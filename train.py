import torch
from tqdm import tqdm
from glob import glob
from data import NpzDataset
from model import UNet
from utils import iou_score
from plot import plot_comparison
import numpy as np
import argparse
import json
import os


def split_data(files, file_fraction, data_splits):
    num_files = file_fraction * len(files)
    train_split, val_split, test_split = data_splits
    train_index = int(train_split * num_files)
    val_index = train_index + int(val_split * num_files)
    test_index = val_index + int(test_split * num_files)
    train_files = files[:train_index]
    val_files = files[train_index:val_index]
    test_files = files[val_index:test_index]
    return train_files, val_files, test_files


def train(
        model, train_loader, device, criterion, optimizer, scheduler, length,
        val_loader, epochs, outdir
    ):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        for data in tqdm(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, not_earth = data['X'].to(device), data['y'].to(device), data['not_earth'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs) # (batch_size, n_fts, img_cols, img_rows)

            outputs = outputs.reshape(length)
            labels = labels.reshape(length)
            not_earth = not_earth.reshape(length)

            # loss = criterion(outputs, labels)
            loss = criterion(outputs[not_earth], labels[not_earth])
            
            loss.backward()
            optimizer.step()

        # print statistics
        print(f'{(epoch + 1):3d} loss: {loss.item()}')
        train_losses.append(loss.item())

        eval_dir = f'{outdir}/{epoch}'
        try:
            os.makedirs(eval_dir)
        except FileExistsError:
            pass
        val_loss = evaluate(model, val_loader, device, criterion, length, eval_dir)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

    losses = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    np.savez(f'{outdir}/losses.npz', **losses)

    return model


def evaluate(model, data_loader, device, criterion, length, outdir):
    model.eval()
    preds = torch.tensor([], dtype=torch.float32).to(device)
    truth = torch.tensor([], dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            inputs, labels, not_earth = data['X'].to(device), data['y'].to(device), data['not_earth'].to(device)

            outputs = unet(inputs)

            results = {
                'outputs': outputs.detach().cpu().numpy().squeeze(),
                'labels': labels.detach().cpu().numpy().squeeze()
            }
            np.savez(f'{outdir}/{i}.npz', **results)

            not_earth = not_earth.reshape(length)
            flat_outputs = outputs.reshape(length)[not_earth]
            flat_labels = labels.reshape(length)[not_earth]

            preds = torch.cat((preds, flat_outputs))
            truth = torch.cat((truth, flat_labels))

            if i == 1:
                plot_comparison(outputs.detach().cpu().numpy().squeeze(), labels.detach().cpu().numpy().squeeze(), f'{outdir}/1.png', i)
    
    loss = criterion(preds, truth)

    print('Test loss:', loss.item())
    
    print('IoU score:', iou_score(truth, preds))

    return loss.item()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--indir', required=True, type=str)
    arg_parser.add_argument('-f', '--file-fraction', default=1.0, type=float)
    arg_parser.add_argument('-d', '--data-splits', default=[0.8, 0.1, 0.1], nargs='+')
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-b', '--batch-size', default=1, type=int)
    arg_parser.add_argument('-l', '--learning-rate', default=1.e-5, type=float)
    arg_parser.add_argument('-c', '--num-classes', default=1, type=int)
    arg_parser.add_argument('-y', '--height', default=344, type=int)
    arg_parser.add_argument('-x', '--width', default=620, type=int)
    arg_parser.add_argument('-n', '--normalize', action='store_true')
    arg_parser.add_argument('-s', '--standardize', action='store_true')
    arg_parser.add_argument('-r', '--raw', action='store_true', help='run with only raw features, otherwise all are used')
    arg_parser.add_argument('-g', '--gpus', nargs='+', help='GPUs to run on in the form 0 1 etc.')
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    files = glob(f'{args.indir}/*.npz')
    train_files, val_files, test_files = split_data(files, args.file_fraction, args.data_splits)
    print(len(train_files), 'train files:', train_files)
    print(len(val_files), 'val files:', val_files)
    print(len(test_files), 'test files:', test_files)

    features = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'vx', 'vy', 'vz', 'rho']
    if not args.raw:
        features += ['anisotropy', 'agyrotropy']

    length = args.batch_size * args.num_classes * args.width * args.height   

    train_dataset = NpzDataset(train_files, features, args.normalize, args.standardize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    val_dataset = NpzDataset(val_files, features, args.normalize, args.standardize)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    test_dataset = NpzDataset(test_files, features, args.normalize, args.standardize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    unet = UNet(
        enc_chs=(len(features), 64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64),
        num_class=args.num_classes,
        retain_dim=True,
        out_sz=(args.height, args.width)
    )
    print(unet)

    if args.gpus:
        assert torch.cuda.is_available()
        device = torch.device(f'cuda:{args.gpus[0]}')
        print('gpus:', args.gpus)
        unet = torch.nn.parallel.DataParallel(unet, device_ids=[int(gpu) for gpu in args.gpus])
    else:
        device = 'cpu'
    
    print('device:', device)
    unet.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=1.e-5)

    unet = train(unet, train_loader, device, criterion, optimizer, scheduler, length, val_loader, args.epochs, args.outdir)
    print('Finished training!\n')

    print('Evaluating...')
    evaluate(unet, test_loader, device, criterion, length, args.outdir)

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


def train(
        model, train_loader, device, criterion, optimizer, scheduler, length,
        test_loader, epochs, outdir
    ):

    train_losses = []
    test_losses = []

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
        test_loss = evaluate(model, test_loader, device, criterion, length, eval_dir)
        test_losses.append(test_loss)

        scheduler.step(test_loss)

    losses = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }

    np.savez(f'{outdir}/losses.npz', **losses)

    return model


def evaluate(model, test_loader, device, criterion, length, outdir):
    model.eval()
    preds = torch.tensor([], dtype=torch.float32).to(device)
    truth = torch.tensor([], dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
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
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-b', '--batch-size', default=1, type=int)
    arg_parser.add_argument('-l', '--learning-rate', default=1.e-5, type=float)
    arg_parser.add_argument('-c', '--num-classes', default=1, type=int)
    arg_parser.add_argument('-h', '--height', default=344, type=int)
    arg_parser.add_argument('-w', '--width', default=620, type=int)
    arg_parser.add_argument('-n', '--normalize', action='store_true')
    arg_parser.add_argument('-s', '--standardize', action='store_true')
    arg_parser.add_argument('-r', '--raw', action='store_true', help='run with only raw features, otherwise all are used')
    arg_parser.add_argument('-g', '--gpus', nargs='+', help='GPUs to run on in the form 0 1 etc.')
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    files = sorted(glob(f'{args.indir}/*.npz'))

    features = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'vx', 'vy', 'vz', 'rho']
    if not args.raw:
        features += ['anisotropy', 'agyrotropy']

    length = args.batch_size * args.num_classes * args.width * args.height

    print(files)
    train_dataset = NpzDataset(files[:20], features, args.normalize, args.standardize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    test_dataset = NpzDataset(files[20:30], features, args.normalize, args.standardize)
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

    unet = train(unet, train_loader, device, criterion, optimizer, scheduler, length, test_loader, args.epochs, args.outdir)
    print('Finished training!\n')

    print('Evaluating...')
    evaluate(unet, test_loader, device, criterion, length, args.outdir)

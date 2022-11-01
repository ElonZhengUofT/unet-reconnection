#!/usr/bin/env python
import torch
from tqdm import tqdm
from glob import glob
from src.data import NpzDataset
from src.model import UNet
from src.utils import EarlyStopping
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
        model, train_loader, device, criterion, optimizer, scheduler, 
        early_stopping, length, val_loader, epochs, outdir
    ):

    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    for epoch in range(1, epochs + 1):
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

            loss = criterion(outputs[not_earth], labels[not_earth])
            
            loss.backward()
            optimizer.step()

        print(f'{epoch:3d} loss: {loss.item()}')
        train_losses.append(loss.item())

        val_dir = os.path.join(outdir, 'val', str(epoch))
        try:
            os.makedirs(val_dir)
        except FileExistsError:
            pass
        val_loss = evaluate(model, val_loader, device, criterion, length, val_dir, epoch)
        val_losses.append(val_loss)
        print('Validation loss:', val_loss)

        if val_loss < best_val_loss:
            state_dict = model.module.state_dict()
            torch.save(state_dict, f=os.path.join(outdir, 'unet.pt'))
            best_model = model
            best_epoch = epoch
            best_val_loss = val_loss

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return best_model, best_epoch, train_losses, val_losses


def evaluate(model, data_loader, device, criterion, length, outdir, epoch):
    model.eval()
    preds = torch.tensor([], dtype=torch.float32).to(device)
    truth = torch.tensor([], dtype=torch.float32).to(device)

    n_dumps = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            inputs, labels, not_earth = data['X'].to(device), data['y'].to(device), data['not_earth'].to(device)

            outputs = unet(inputs)

            not_earth = not_earth.reshape(length)
            flat_outputs = outputs.reshape(length)[not_earth]
            flat_labels = labels.reshape(length)[not_earth]

            preds = torch.cat((preds, flat_outputs))
            truth = torch.cat((truth, flat_labels))

            if n_dumps < 5:
                for c in range(outputs.shape[0]):
                    plot_comparison(
                        preds=outputs[c].detach().cpu().numpy().squeeze(), 
                        truth=labels[c].detach().cpu().numpy().squeeze(), 
                        file=os.path.join(outdir, f'{i}.png'),
                        epoch=epoch
                    )

                    results = {
                        'outputs': outputs.detach().cpu().numpy().squeeze(),
                        'labels': labels.detach().cpu().numpy().squeeze()
                    }
                    np.savez(os.path.join(outdir, f'{i}.npz'), **results)

                    n_dumps += 1
                    if n_dumps == 5:
                        break
    
    loss = criterion(preds, truth)
    return loss.item()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--indir', required=True, type=str)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-f', '--file-fraction', default=1.0, type=float)
    arg_parser.add_argument('-d', '--data-splits', default=[0.8, 0.1, 0.1], nargs='+')
    arg_parser.add_argument('-e', '--epochs', default=10, type=int)
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

    files = glob(os.path.join(args.indir, '*.npz'))
    train_files, val_files, test_files = split_data(files, args.file_fraction, args.data_splits)
    print(len(train_files), 'train files:', train_files)
    print(len(val_files), 'val files:', val_files)
    print(len(test_files), 'test files:', test_files)

    features = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'vx', 'vy', 'vz', 'rho']
    if not args.raw:
        features += ['anisotropy', 'agyrotropy']
    print(len(features), 'features:', features)

    length = args.batch_size * args.num_classes * args.width * args.height   

    train_dataset = NpzDataset(train_files, features, args.normalize, args.standardize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    val_dataset = NpzDataset(val_files, features, args.normalize, args.standardize)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)

    test_dataset = NpzDataset(test_files, features, args.normalize, args.standardize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    unet = UNet(
        enc_chs=(len(features), 64, 128, 256),
        dec_chs=(256, 128, 64),
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1.e-5, verbose=True
    )
    early_stopping = EarlyStopping()

    best_model, best_epoch, train_losses, val_losses = train(
        unet, train_loader, device, criterion, optimizer, scheduler, 
        early_stopping, length, val_loader, args.epochs, args.outdir
    )
    print('Finished training!')

    print('Evaluating best model from epoch', best_epoch)
    test_dir = os.path.join(args.outdir, 'test')
    try:
        os.makedirs(test_dir)
    except FileExistsError:
        pass
    test_loss = evaluate(best_model, test_loader, device, criterion, length, test_dir, args.epochs)
    print('Test loss:', test_loss)

    with open(os.path.join(args.outdir, 'metadata.json'), 'w') as f:
        json.dump(
            {
                'args': vars(args),
                'features': features,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_loss': test_loss,
                'best_epoch': best_epoch,
                'train_files': train_files,
                'val_files': val_files,
                'test_files': test_files,
            },
            fp=f,
            indent=2
        )

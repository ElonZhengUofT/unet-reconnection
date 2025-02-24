#!/usr/bin/env python
import torch
from tqdm import tqdm
from glob import glob
from src.data import NpzDataset
from src.model import UNet
from src.NewUnet import UNet as NewUNet
from src.callbacks import EarlyStopping
from src.utils import split_data
from plot import plot_comparison
from ptflops import get_model_complexity_info
import numpy as np
import argparse
import json
import os


def train(
        model, train_loader, device, criterion, optimizer, scheduler, 
        early_stopping, val_loader, epochs, lr, binary, outdir
    ):

    train_losses = []
    val_losses = []
    lr_change_epoch = 0
    lr_history = {lr_change_epoch: lr}
    best_val_loss = np.inf
    
    for epoch in range(1, epochs + 1):
        total_count = 0
        total_loss = 0
        print('Epoch:', epoch)
        with tqdm(train_loader) as tq:
            for data in tq:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, not_earth = (
                    data['X'].to(device), data['y'].to(device), 
                    data['not_earth'].to(device)
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs) # (batch_size, n_classes, img_cols, img_rows)           

                loss = criterion(outputs[not_earth], labels[not_earth])
                
                loss.backward()
                optimizer.step()

                loss = loss.item()
                total_loss += loss

                count = labels.size(0)
                total_count += count

                tq.set_postfix({
                    'Loss': '%.7f' % (loss / count)
                })

        print(f'Training loss: {total_loss / total_count}')
        train_losses.append(total_loss / total_count)

        val_dir = os.path.join(outdir, 'val', str(epoch))
        os.makedirs(val_dir, exist_ok=True)

        val_loss = evaluate(
            model, val_loader, device, criterion, 
            val_dir, epoch, binary, mode='val'
        )
        val_losses.append(val_loss)
        print('Validation loss:', val_loss)

        if val_loss < best_val_loss:
            if args.gpus:
                torch.save(model.module.state_dict(), f=os.path.join(outdir, 'unet_best_epoch.pt'))
            else:
                torch.save(model.state_dict(), f=os.path.join(outdir, 'unet_best_epoch.pt'))
            torch.save(optimizer.state_dict(), f=os.path.join(outdir, 'optimizer_best_epoch.pt'))
            best_model = model
            best_epoch = epoch
            best_val_loss = val_loss

        scheduler.step(val_loss)
        early_stopping(val_loss)

        [last_lr] = scheduler._last_lr
        if last_lr < lr_history[lr_change_epoch]:
            # Store which epochs lr changes at
            lr_change_epoch = epoch
            lr_history[lr_change_epoch] = last_lr
            
            print('Restoring best model weights.')
            if args.gpus:
                model.module.load_state_dict(
                    torch.load(
                        os.path.join(outdir, 'unet_best_epoch.pt'), map_location=device
                    )
                )
            else:
                model.load_state_dict(
                    torch.load(
                        os.path.join(outdir, 'unet_best_epoch.pt'), map_location=device
                    )
                )
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(outdir, 'optimizer_best_epoch.pt'), map_location='cpu'
                )
            )

        if early_stopping.early_stop:
            break

    return best_model, best_epoch, epoch, lr_history, train_losses, val_losses


def evaluate(model, data_loader, device, criterion, outdir, epoch, binary, mode):
    model.eval()

    total_loss = 0
    correct = 0
    total_correct = 0
    count = 0
    total_count = 0

    with torch.no_grad():
        with tqdm(enumerate(data_loader)) as tq:
            for i, data in tq:
                inputs, labels, not_earth, fname = (
                    data['X'].to(device), data['y'].to(device), 
                    data['not_earth'].to(device), data['fname']
                )

                outputs = model(inputs)

                loss = criterion(outputs[not_earth], labels[not_earth]).item()
                
                total_loss += loss

                batch_size = labels.size(0)
                width = labels.size(2)
                height = labels.size(3)
                count = batch_size
                total_count += count

                if binary:
                    threshold_outputs = torch.where(outputs > 0.5, 1, 0)
                    correct = (threshold_outputs == labels[:, 0]).sum().item()
                else:
                    _, outputs = outputs.max(1) # choose the most likely class
                    correct = (outputs == labels[:, 0]).sum().item()

                if (mode == 'test') or (i == 0):
                    if mode == 'val':
                        num_plots = 1
                    else:
                        num_plots = batch_size
                    for n in range(num_plots):
                        preds_np = outputs[n].detach().cpu().numpy().squeeze()
                        truth_np = labels[n, 0].detach().cpu().numpy().squeeze()

                        plot_comparison(
                            preds=preds_np,
                            truth=truth_np, 
                            file=os.path.join(outdir, f'{fname[n]}.png'),
                            epoch=epoch
                        )

                        results = {
                            'outputs': preds_np,
                            'labels': truth_np
                        }
                        np.savez(os.path.join(outdir, f'{fname[n]}.npz'), **results)

                tq.set_postfix({
                    'Loss': '%.7f' % (loss / count),
                    'Accuracy': '%.7f' % (correct / (batch_size * width * height)),
                })

    return total_loss / total_count


if __name__ == '__main__':
    os.environ["PYTHONPATH"] = "/content/unet-reconnection"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--indir', required=True, type=str)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-f', '--file-fraction', default=1.0, type=float)
    arg_parser.add_argument('-d', '--data-splits', default=[0.8, 0.1, 0.1], nargs='+', type=float)
    arg_parser.add_argument('-e', '--epochs', default=10, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=1, type=int)
    arg_parser.add_argument('-l', '--learning-rate', default=1.e-5, type=float)
    arg_parser.add_argument('-c', '--num-classes', default=1, type=int)
    arg_parser.add_argument('-k', '--kernel-size', default=3, type=int)
    arg_parser.add_argument('-y', '--height', default=344, type=int)
    arg_parser.add_argument('-x', '--width', default=620, type=int)
    arg_parser.add_argument('-n', '--normalize', action='store_true')
    arg_parser.add_argument('-s', '--standardize', action='store_true')
    arg_parser.add_argument('-g', '--gpus', nargs='+', help='GPUs to run on in the form 0 1 etc.')
    arg_parser.add_argument('-w', '--num-workers', default=0, type=int)
    arg_parser.add_argument('--velocity', action='store_true')
    arg_parser.add_argument('--rho', action='store_true')
    arg_parser.add_argument('--anisotropy', action='store_true')
    arg_parser.add_argument('--agyrotropy', action='store_true')
    args = arg_parser.parse_args()

    print("First Checkpoint")

    os.makedirs(args.outdir, exist_ok=True)

    files = glob(os.path.join(args.indir, '*.npz'))
    train_files, val_files, test_files = split_data(files, args.file_fraction, args.data_splits)
    print(len(train_files), 'train files:', train_files)
    print(len(val_files), 'val files:', val_files)
    print(len(test_files), 'test files:', test_files)

    features = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
    if args.velocity:
        features += ['vx', 'vy', 'vz']
    if args.rho:
        features += ['rho']
    if args.anisotropy:
        features += ['anisotropy']
    if args.agyrotropy:
        features += ['agyrotropy']
    print(len(features), 'features:', features)

    binary = args.num_classes == 1

    train_dataset = NpzDataset(train_files, features, args.normalize, args.standardize, binary)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)


    val_dataset = NpzDataset(val_files, features, args.normalize, args.standardize, binary)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)

    test_dataset = NpzDataset(test_files, features, args.normalize, args.standardize, binary)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)

    print("Second Checkpoint")

    # unet = UNet(enc_chs=(len(features), 64, 128),
    #             dec_chs=(128, 64),
    #             num_class=args.num_classes,
    #             retain_dim=True,
    #             out_sz=(args.height, args.width),
    #             kernel_size=args.kernel_size
    # )

    unet = NewUNet(
        down_chs=(len(features), 64, 128, 256),
        up_chs=(256, 128, 64),
        num_class=args.num_classes,
        retain_dim=True,
        out_sz=(args.height, args.width),
        kernel_size=args.kernel_size
    )

    print("Third Checkpoint")
    
    macs, params = get_model_complexity_info(
        unet, (len(features), args.height, args.width), 
        as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if args.gpus:
        assert torch.cuda.is_available()
        device = torch.device(f'cuda:{args.gpus[0]}')
        print('gpus:', args.gpus)
        unet = torch.nn.parallel.DataParallel(unet, device_ids=[int(gpu) for gpu in args.gpus])
    else:
        device = 'cpu'
    
    print('device:', device)
    unet.to(device)

    if args.num_classes == 1:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1.e-5, verbose=True
    )
    early_stopping = EarlyStopping(patience=10, min_delta=0)

    print('Starting training...')

    best_model, best_epoch, last_epoch, lr_history, train_losses, val_losses = train(
        unet, train_loader, device, criterion, optimizer, scheduler, early_stopping, 
        val_loader, args.epochs, args.learning_rate, binary, args.outdir
    )
    print('Finished training!')

    print('Evaluating best model from epoch', best_epoch)
    test_dir = os.path.join(args.outdir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    test_loss = evaluate(
        best_model, test_loader, device, criterion, 
        test_dir, best_epoch, binary, mode='test'
    )
    print('Test loss:', test_loss)

    with open(os.path.join(args.outdir, 'metadata.json'), 'w') as f:
        json.dump(
            {
                'args': vars(args),
                'features': features,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_loss': test_loss,
                'last_epoch': last_epoch,
                'best_epoch': best_epoch,
                'lr_history': lr_history,
                'train_files': train_files,
                'val_files': val_files,
                'test_files': test_files,
            },
            fp=f,
            indent=2
        )

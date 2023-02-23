#!/usr/bin/env python
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from src.data import NpzDataset
from src.model import UNet
from ptflops import get_model_complexity_info
import time


def predict(model, data_loader, device, criterion, outdir):
    model.eval()

    total_loss = 0
    count = 0
    total_count = 0

    inference_times = []

    with torch.no_grad():
        with tqdm(enumerate(data_loader)) as tq:
            for i, data in tq:
                inputs, labels, not_earth, fname = (
                    data['X'].to(device), data['y'].to(device), 
                    data['not_earth'].to(device), data['fname']
                )

                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()

                inference_times.append(end_time - start_time)

                loss = criterion(outputs[not_earth], labels[not_earth]).item()
                total_loss += loss

                batch_size = labels.size(0)
                width = labels.size(2)
                height = labels.size(3)
                count = batch_size
                total_count += count

                for n in range(batch_size):
                    preds_np = outputs[n].detach().cpu().numpy().squeeze()
                    truth_np = labels[n, 0].detach().cpu().numpy().squeeze()

                    results = {
                        'outputs': preds_np,
                        'labels': truth_np
                    }
                    np.savez(os.path.join(outdir, f'{fname[n]}.npz'), **results)

                tq.set_postfix({
                    'Loss': '%.7f' % (loss / count)
                })

    return total_loss / total_count, inference_times


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--indir', required=True, type=str)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-w', '--num-workers', default=0, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=1, type=int)
    arg_parser.add_argument('-g', '--gpus', nargs='+', help='GPUs to run on in the form 0 1 etc.')
    args = arg_parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.indir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        meta_args = metadata['args']

    features = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
    if meta_args['velocity']:
        features += ['vx', 'vy', 'vz']
    if meta_args['rho']:
        features += ['rho']
    if meta_args['anisotropy']:
        features += ['anisotropy']
    if meta_args['agyrotropy']:
        features += ['agyrotropy']

    binary = meta_args['num_classes'] == 1
    test_dataset = NpzDataset(metadata['test_files'], features, meta_args['normalize'], meta_args['standardize'], binary)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)

    unet = UNet(
        enc_chs=(len(features), 64, 128, 256),
        dec_chs=(256, 128, 64),
        num_class=meta_args['num_classes'],
        retain_dim=True,
        out_sz=(meta_args['height'], meta_args['width'])
    )

    macs, params = get_model_complexity_info(
        unet, (len(features), meta_args['height'], meta_args['width']), 
        as_strings=True,
        print_per_layer_stat=True, verbose=True
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

    if binary:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=meta_args['learning_rate'])

    if args.gpus:
        unet.module.load_state_dict(
            torch.load(
                os.path.join(args.indir, 'unet_best_epoch.pt'), map_location=device
            )
        )
    else:
        unet.load_state_dict(
            torch.load(
                os.path.join(args.indir, 'unet_best_epoch.pt'), map_location=device
            )
        )

    test_loss, inference_times = predict(unet, test_loader, device, criterion, args.outdir)

    print(inference_times[1:])
    print('Mean inference time:', np.mean(inference_times[1:])) 
    print('Stdev inference time:', np.std(inference_times[1:]))

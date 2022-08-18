import torch
from tqdm import tqdm
from glob import glob
from data import NpzDataset
from model import UNet
import numpy as np
import argparse
import os


def train(model, train_loader, device, criterion, optimizer, length):
    for epoch in range(epochs):
        for data in tqdm(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['X'].to(device), data['y'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs) # (batch_size, n_classes, img_cols, img_rows)

            outputs = outputs.reshape(length)
            labels = labels.reshape(length)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        print(f'{(epoch + 1):3d} loss: {loss.item()}')

    return model


def evaluate(model, test_loader, device, criterion, length, outdir):
    model.eval()
    preds = torch.tensor([], dtype=torch.float32).to(device)
    truth = torch.tensor([], dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            inputs, labels = data['X'].to(device), data['y'].to(device)

            outputs = unet(inputs)

            results = {
                'outputs': outputs.detach().cpu().numpy().squeeze(),
                'labels': labels.detach().cpu().numpy().squeeze()
            }
            np.savez(f'{outdir}/{i}.npz', **results)

            flat_outputs = outputs.reshape(length)
            flat_labels = labels.reshape(length)

            preds = torch.cat((preds, flat_outputs))
            truth = torch.cat((truth, flat_labels))
    
    loss = criterion(preds, truth)

    print(f'Test loss: {loss.item()}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    files = glob('data/*.npz')
    features = ['Bx', 'By', 'Bz', 'rho', 'anisoP', 'agyrotropy', 'absE']
    height_out = 216
    width_out = 535
    num_classes = 1
    batch_size = 1
    epochs = args.epochs

    length = batch_size * num_classes * width_out * height_out

    train_dataset = NpzDataset(files[:12], features)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = NpzDataset(files[12:], features)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    unet = UNet(
        enc_chs=(len(features), 64, 128),
        dec_chs=(128, 64),
        num_class=1,
        retain_dim=True,
        out_sz=(height_out, width_out)
    ).to(device)
    print(unet)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)

    unet = train(unet, train_loader, device, criterion, optimizer, length)
    print('Finished training!\n')

    print('Evaluating...')
    evaluate(unet, test_loader, device, criterion, length, args.outdir)

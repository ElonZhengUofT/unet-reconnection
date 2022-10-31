#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from src.utils import iou_score
import argparse
import json
import gif
import os


def plot_reconnection_points(file):
    data = np.load(file)
    fig = plt.figure(figsize=(10, 6))

    xx = np.linspace(data['xmin'], data['xmax'], (data['anisotropy'].shape[1]))
    zz = np.linspace(data['zmin'], data['zmax'], (data['anisotropy'].shape[0]))

    labeled_indices = data['labeled_domain'].nonzero()
    labeled_z = zz[labeled_indices[0]]
    labeled_x = xx[labeled_indices[1]]

    ax = fig.add_subplot()

    c = ax.imshow(data['anisotropy'], extent=[data['xmin'], data['xmax'], data['zmin'], data['zmax']])
    ax.scatter(labeled_x, labeled_z, marker='x', color='red')
    ax.set_title('Pseudocolor-Anisotropy with reconnection points', fontsize=16)
    ax.set_xlabel('x/Re', fontsize=12)
    ax.set_ylabel('z/Re', fontsize=12)
    fig.colorbar(c, ax=ax)

    fig.savefig('reconnection_points.png', bbox_inches='tight')
    plt.close()


@gif.frame
def plot_comparison(preds, truth, file, epoch):
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    c1 = ax1.imshow(preds)
    fig.colorbar(c1, ax=ax1)
    ax1.set_title(f'Preds, epoch {epoch}')

    c2 = ax2.imshow(truth)
    fig.colorbar(c2, ax=ax2)
    ax2.set_title('Truth')

    plt.savefig(file)


def plot_loss(train_losses, val_losses, outdir):
    x = range(2, len(train_losses) + 1)
    plt.plot(x, train_losses[1:], label='Training loss')
    plt.plot(x, val_losses[1:], label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss_zoom.png'))
    plt.close()


def plot_iou_score(epochs, iou_scores, outdir):
    plt.plot(range(epochs), iou_scores)
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.savefig(os.path.join(outdir, 'iou_score.png'))
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True, type=str)
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    args = arg_parser.parse_args()

    files = glob(os.path.join(args.dir, '*.npz'))

    plot_reconnection_points('data/3600.npz')

    with open(os.path.join(args.dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    plot_loss(metadata['train_losses'], metadata['val_losses'], args.dir)

    frames = []
    iou_scores = []
    for i in range(args.epochs):
        data = np.load(os.path.join(args.dir, 'val', str(i), '0.npz'))
        preds, truth = data['outputs'], data['labels']
        iou_scores.append(iou_score(truth, preds))
        frame = plot_comparison(preds, truth, os.path.join(args.dir, 'val', str(i), '0.png'), i)
        frames.append(frame)

    gif.save(frames, os.path.join(args.dir, 'epochs.gif'), duration=100)
    plot_iou_score(args.epochs, iou_scores, args.dir)

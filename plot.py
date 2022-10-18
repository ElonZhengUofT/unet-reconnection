import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse
import gif
import os


def plot_reconnection_points(file):
    data = np.load(file)
    print(data.files)

    xmin = data['xmin']
    xmax = data['xmax']
    zmin = data['zmin']
    zmax = data['zmax']
    anisoP = data['anisoP']
    labeled_domain = data['labeled_domain']

    xx = np.linspace(xmin, xmax, (np.array(anisoP).shape[1]))
    zz = np.linspace(zmin, zmax, (np.array(anisoP).shape[0]))

    labeled_indices = labeled_domain.nonzero()
    labeled_z = zz[labeled_indices[0]]
    labeled_x = xx[labeled_indices[1]]

    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = '14'

    ax1= fig.add_subplot(2, 1, 1)
    ax2= fig.add_subplot(2, 1, 2)

    c1 = ax1.pcolor(xx, zz, anisoP)
    c1 = ax1.scatter(labeled_x, labeled_z, marker='x', color='red', s=134)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([zmin, zmax])
    ax1.title.set_text('Pseudocolor-Anisotropy; red crosses - X-points')
    fig.colorbar(c1, ax=ax1)

    c2 = ax2.pcolor(xx, zz, labeled_domain)
    ax2.title.set_text('Labeled pixels')
    ax2.set_xlabel('x/Re',fontsize=16)
    ax2.set_ylabel('z/Re',fontsize=16)
    fig.colorbar(c2, ax=ax2)

    plt.savefig('reconnection_points.png')


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


def plot_loss(losses, outdir):
    x = range(21, len(losses['train_losses']) + 1)
    plt.plot(x, losses['train_losses'][20:], label='Training loss')
    plt.plot(x, losses['test_losses'][20:], label='Test loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss_zoom.png'))
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True, type=str)
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    args = arg_parser.parse_args()

    files = glob(f'{args.dir}/*.npz')

    # plot_reconnection_points('data-large/3612.npz')

    losses = np.load(f'{args.dir}/losses.npz')
    plot_loss(losses, args.dir)


    # first_file = np.load('data-large/3600.npz')
    # earth = first_file['rho'] == 0

    frames = []
    for i in range(args.epochs):
        data = np.load(f'{args.dir}/{i}/0.npz')
        preds, truth = data['outputs'], data['labels']

        # preds[earth] = 0

        frame = plot_comparison(preds, truth, f'{args.dir}/{i}/0.png', i)
        frames.append(frame)

    gif.save(frames, f'{args.dir}/epochs.gif', duration=100)
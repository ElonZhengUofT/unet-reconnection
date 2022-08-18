import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse


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


def plot_comparison(preds, truth, file):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    c1 = ax1.imshow(preds)
    fig.colorbar(c1, ax=ax1)
    ax1.set_title('Preds')

    c2 = ax2.imshow(truth)
    fig.colorbar(c2, ax=ax2)
    ax2.set_title('Truth')

    plt.savefig(file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True)
    args = arg_parser.parse_args()

    files = glob(f'{args.dir}/*.npz')

    for i, file in enumerate(files):
        data = np.load(file)
        preds, truth = data['outputs'], data['labels']
        plot_comparison(preds, truth, f'{args.dir}/{i}.png')

#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from glob import glob
import src.utils as utils
from src.utils import iou_score, pick_best_threshold
from sklearn import metrics
from pathlib import Path
import argparse
import json
import gif
import os
import sys


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
    
    earth_center_x = np.argmin(np.abs(xx)) # center of earth x coord

    return earth_center_x, data['xmin'], data['xmax'], data['zmin'], data['zmax']


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


def generate_geom_seq(num_epochs):
    seq = [1]
    i = 1
    step = 1
    while True:
        if i % 10 == 0:
            step += 1
        seq.append(seq[-1] + step)
        if seq[-1] >= num_epochs:
            break
        i += 1
    
    if seq[-1] > num_epochs:
        del seq[-1]

    return seq


@gif.frame
def plot_gif_frame(preds, truth, epoch, xmin, xmax, zmin, zmax):
    fig = plt.figure(figsize=(5, 3), dpi=100)

    xx = np.linspace(xmin, xmax, (truth.shape[1]))
    zz = np.linspace(zmin, zmax, (truth.shape[0]))

    labeled_indices = truth.nonzero()
    labeled_z = zz[labeled_indices[0]]
    labeled_x = xx[labeled_indices[1]]

    ax = fig.add_subplot()

    c = ax.imshow(preds, extent=[xmin, xmax, zmin, zmax])
    # ax.scatter(labeled_x, labeled_z, marker='x', color='red', s=50)
    ax.set_title(f'Epoch {epoch}')
    ax.set_xlabel('x/Re')
    ax.set_ylabel('z/Re')
    plt.tight_layout()
    # fig.colorbar(c, ax=ax)


def plot_loss(train_losses, val_losses, lr_history, outdir):
    x = range(4, len(train_losses) + 1)
    
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(fmt)
    
    plt.plot(x, train_losses[3:], label='Training loss')
    plt.plot(x, val_losses[3:], label='Validation loss')

    if lr_history:
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(lr_history[1:], ymin=ymin, ymax=ymax, ls='dashed', lw=0.8, colors='gray')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss_zoom.png'))
    plt.close()


def plot_roc(preds, truth, outdir):
    fpr, tpr, threshold = metrics.roc_curve(truth, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:0.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(outdir, 'roc_curve.png'))
    plt.close()


def plot_precision_recall(
        precision, recall, 
        max_f1_score, max_f1_index, max_f1_thresh, 
        max_f2_score, max_f2_index, max_f2_thresh, 
        outdir
    ):
    plt.title('Precision Recall')
    plt.plot(recall, precision, marker='.', markersize=2, label='U-Net')
    plt.plot(
        recall[max_f1_index], precision[max_f1_index], marker='.', color='tab:green', markersize=12, 
        label=f'Max F1 = {max_f1_score:.4f}\nThreshold = {max_f1_thresh:.4f}'
    )
    plt.plot(
        recall[max_f2_index], precision[max_f2_index], marker='^', color='tab:red', markersize=7, 
        label=f'Max F2 = {max_f2_score:.4f}\nThreshold = {max_f2_thresh:.4f}'
    )
    plt.legend()
    plt.minorticks_on()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tick_params(axis='both', which='both', direction='in')
    plt.savefig(os.path.join(outdir, 'precision_recall.png'))
    plt.close()


def plot_thresholds(precision, recall, thresholds, max_f1_thresh, max_f2_thresh, outdir):
    plt.title('Precision & Recall with Different Thresholds')
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    # plt.plot(thresholds[::10], [utils.iou_score(np.where(preds < t, 0, 1), truth) for t in thresholds[::10]], label='IoU')
    # ymin, ymax = plt.gca().get_ylim()
    plt.axvline(max_f1_thresh, ymin=0.04, ymax=0.96, ls='--', c='gray', label='Max F1 Threshold')
    plt.axvline(max_f2_thresh, ymin=0.04, ymax=0.96, ls='-.', c='black', label='Max F2 Threshold')
    plt.legend()
    plt.minorticks_on()
    plt.ylabel('Score')
    plt.xlabel('Decision Threshold')
    plt.tick_params(axis='both', which='both', direction='in')
    plt.savefig(os.path.join(outdir, 'thresholds.png'))
    plt.close()


def plot_confusion_matrix(binary_preds, truth, score, outdir):
    plt.title('Confusion Matrix')
    cm = metrics.confusion_matrix(truth, binary_preds).T
    plt.imshow(cm, norm=mpl.colors.LogNorm())
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for i in [0, 1]:
        for j in [0, 1]:
            text_color = 'yellow' if cm[i, j] < np.max(cm) / 2 else 'black'
            plt.annotate(cm[i, j], (i, j), va='center', ha='center', color=text_color)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, f'confusion_matrix_{score}.png'))
    plt.close()


def evaluate_classifier(preds, truth):
    tn, fp, fn, tp = metrics.confusion_matrix(truth, preds).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    auc_roc = metrics.roc_auc_score(y_score=preds, y_true=truth)
    iou = utils.iou_score(preds, truth)

    return {
        'True Positive' : int(tp),
        'True Negative' : int(tn),
        'False Positive': int(fp),
        'False Negative': int(fn),
        'Accuracy' : accuracy,
        'Precision': precision,
        'Recall/Sensitivity' : recall,
        'Specificity': tn / (fp + tn),
        'AUC ROC': auc_roc,
        'IoU': iou
    }


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True, type=str)
    arg_parser.add_argument('-g', '--gif', action='store_true')
    args = arg_parser.parse_args()
    
    # Read metadata
    with open(os.path.join(args.dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Plot loss curve
    if 'lr_history' in metadata.keys():
        lr_history = [int(epoch) for epoch in metadata['lr_history'].keys()]
    else:
        lr_history = None
    plot_loss(metadata['train_losses'], metadata['val_losses'], lr_history, args.dir)

    # Plot anisotropy with reconnection points
    earth_center_x, xmin, xmax, zmin, zmax = plot_reconnection_points('sample/data/3600.npz')

    if args.gif:
        # Create animation of validation predictions
        frames = []
        fname = Path(metadata['val_files'][0]).stem
        sequence = generate_geom_seq(metadata['last_epoch'])
        for i in sequence:
            data = np.load(os.path.join(args.dir, 'val', str(i), f'{fname}.npz'))
            preds, truth = data['outputs'], data['labels']
            frame = plot_gif_frame(preds, truth, i, xmin, xmax, zmin, zmax)
            frames.append(frame)
        gif.save(frames, os.path.join(args.dir, 'epochs.gif'), duration=100)

    # Load test predictions
    test_list = glob(os.path.join(args.dir, 'test', '*.npz'))
    num_test_files = len(test_list)
    all_preds = np.zeros((num_test_files, metadata['args']['height'], metadata['args']['width']))
    all_truth = np.zeros((num_test_files, metadata['args']['height'], metadata['args']['width']))
    for i, test_file in enumerate(test_list):
        data = np.load(test_file)
        preds, truth = data['outputs'], data['labels']
        all_preds[i] = preds
        all_truth[i] = truth

    nightside_preds = all_preds[:,:,:earth_center_x]
    nightside_truth = all_truth[:,:,:earth_center_x]

    dayside_preds = all_preds[:,:,earth_center_x:]
    dayside_truth = all_truth[:,:,earth_center_x:]

    f1 = {}
    f2 = {}
    for preds, truth, side in [
        (all_preds.ravel(), all_truth.ravel(), 'both_sides'),
        (nightside_preds.ravel(), nightside_truth.ravel(), 'nightside'),
        (dayside_preds.ravel(), dayside_truth.ravel(), 'dayside')
    ]:
        side_dir = os.path.join(args.dir, side)
        os.makedirs(side_dir, exist_ok=True)

        precision, recall, thresholds = metrics.precision_recall_curve(truth, preds)

        max_f1_score, max_f1_index, max_f1_thresh = utils.pick_best_threshold(precision, recall, thresholds, 1)
        f1[side] = {'score': max_f1_score, 'threshold': max_f1_thresh}
        binary_preds = np.where(preds < max_f1_thresh, 0, 1)
        plot_confusion_matrix(binary_preds, truth, 'f1', side_dir)

        max_f2_score, max_f2_index, max_f2_thresh = utils.pick_best_threshold(precision, recall, thresholds, 2)
        f2[side] = {'score': max_f2_score, 'threshold': max_f2_thresh}
        binary_preds = np.where(preds < max_f2_thresh, 0, 1)
        plot_confusion_matrix(binary_preds, truth, 'f2', side_dir)

        plot_precision_recall(
            precision, recall, 
            max_f1_score, max_f1_index, max_f1_thresh, 
            max_f2_score, max_f2_index, max_f2_thresh, 
            side_dir
        )
        plot_thresholds(precision, recall, thresholds, max_f1_thresh, max_f2_thresh, side_dir)

    all_binary_preds = np.concatenate((
        np.where(nightside_preds < f1['nightside']['threshold'], 0, 1),
        np.where(dayside_preds < f1['dayside']['threshold'], 0, 1)),
        axis=2
    )
    plot_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(), 'f1', args.dir)

    all_binary_preds = np.concatenate((
        np.where(nightside_preds < f2['nightside']['threshold'], 0, 1),
        np.where(dayside_preds < f2['dayside']['threshold'], 0, 1)),
        axis=2
    )
    plot_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(), 'f2', args.dir)
    
    # Plot ROC curve
    plot_roc(all_preds.ravel(), all_truth.ravel(), args.dir)

    metrics = evaluate_classifier(all_binary_preds.ravel(), all_truth.ravel())
    metrics['F1'] = f1
    metrics['F2'] = f2
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, fp=f, indent=2)

#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def compare_precision_recall(data, side):
    _, ax = plt.subplots()

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1, 10000)
        y = f_score * x / (2 * x - f_score) # F1
        # y = f_score * x / (5 * x - 4 * f_score) # F2
        (l,) = plt.plot(x[(y >= 0) & (y <= 1)], y[(y >= 0) & (y <= 1)], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[9900] + 0.02))

    for run in ['em-normalized-3', 'raw-normalized-3', 'custom-normalized-3']:
        auc = metrics.auc(data[run]['recall'], data[run]['precision'])
        ax.plot(data[run]['recall'], data[run]['precision'], label=f'{run} (area = {auc:.2f})')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(['iso-f1 curves'])
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(handles=handles, labels=labels, loc='upper right')
    ax.set_title(side)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.tick_params(axis='both', direction='in')

    plt.savefig(os.path.join('jobviz', f'precision_recall_{side}.png'))


precision_recall = {}
precision_recall['dayside'] = {}
precision_recall['nightside'] = {}
for run in ['em-normalized-3', 'raw-normalized-3', 'custom-normalized-3']:
    precision_recall['dayside'][run] = np.load(os.path.join('wrapped_results', run, 'dayside', 'precision_recall.npz'))
    precision_recall['nightside'][run] = np.load(os.path.join('wrapped_results', run, 'nightside', 'precision_recall.npz'))

compare_precision_recall(precision_recall['dayside'], 'dayside')
compare_precision_recall(precision_recall['nightside'], 'nightside')

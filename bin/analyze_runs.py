#!/usr/bin/env python

import os
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_pivoted_table(variable, folder):
    # define the feature sets, preprocessing methods, and kernel sizes
    feature_set = ['em', 'raw', 'custom']
    preprocessing = ['none', 'normalized', 'standardized']
    kernel_sizes = [1, 3, 5, 7, 9]

    # initialize an empty dictionary to hold the test losses
    values = {}

    # iterate over all possible combinations of feature set, preprocessing, and kernel size
    for f, p, k in itertools.product(feature_set, preprocessing, kernel_sizes):
        job_name = f'{f}-{p}-{k}'
        metadata_file = os.path.join('results', job_name, 'metadata.json')
        
        # if metadata file exists, read the test loss and store it in the values dictionary
        if os.path.isfile(metadata_file):
            with open(metadata_file, 'r') as json_file:
                metadata = json.load(json_file)
                test_loss = float(metadata[variable])
                col_name = f'{f}-{p}'
                values[(k, col_name)] = test_loss

    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame.from_dict(values, orient='index', columns=[variable])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['kernel_size', 'feature_set'])

    # pivot table to a format fit for a heatmap
    df = pd.pivot_table(df, values=variable, index='kernel_size', columns='feature_set')
    df.to_csv(os.path.join(folder, f'{variable}.csv'), float_format='%.4e')     

    return df


def plot_heatmap(df, variable, folder):
    # plot the dataframe as a heatmap
    plt.figure(figsize=(8, 6))
    plt.title(variable)
    plt.imshow(df, cmap='coolwarm')
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
    plt.yticks(range(len(df.index)), df.index)
    plt.xlabel('Feature set - Preprocessing')
    plt.ylabel('Kernel size')
    plt.colorbar()
    plt.tight_layout()

    # save the figure as a png
    plt.savefig(os.path.join(folder, f'{variable}.png'), dpi=300)
    plt.close()


def plot_correlation(df_epochs, df_losses, folder):
    # df_losses.loc[0, 'em-none'] = 1

    # set up the plot
    fig, ax = plt.subplots()

    # plot each loss curve as a scatter plot
    custom_df = df_losses.copy()
    custom_df[custom_df > 1.e-4] = np.nan
    for col in df_losses.columns[1:]:
        if col == 'raw-none' or col == 'custom-none':
            continue
        ax.scatter(df_epochs[col], custom_df[col], label=col, s=7*df_epochs.index)

    # fit a straight line to the data using numpy
    x = df_epochs.values[:, 1:].flatten()
    y = df_losses.values[:, 1:].flatten()

    mask = y < 1.e-4
    x = x[mask]
    y = y[mask]

    slope, intercept = np.polyfit(x, y, 1)

    # plot the fitted line and add equation to the legend
    # line_label = f'y={slope:.2e}x+{intercept:.2e}'
    # ax.plot(x, slope*x + intercept, label=line_label, color='black', linewidth=.7)

    # set axis labels and legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.legend()

    # calculate and print the Pearson correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    print(f'Pearson correlation: {corr:.3f}')

    plt.title(f'Pearson correlation: {corr:.2f}')

    # show the plot
    plt.savefig(os.path.join(folder, 'correlation.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', default='jobviz', type=str)
    args = arg_parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    df_epochs = make_pivoted_table('best_epoch', args.dir)
    df_losses = make_pivoted_table('test_loss', args.dir)

    mask = df_losses > 1e-4

    df_losses[mask] = np.nan
    df_epochs[mask] = np.nan

    plot_heatmap(df_epochs, 'best_epoch', args.dir)
    plot_heatmap(df_losses, 'test_loss', args.dir)

    plot_correlation(df_epochs, df_losses, args.dir)

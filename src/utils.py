import torch
import numpy as np


def iou_score(target, prediction):
    if torch.is_tensor(target) and torch.is_tensor(prediction):
        intersection = torch.logical_and(target, prediction)
        union = torch.logical_or(target, prediction)
        iou_score = torch.sum(intersection) / torch.sum(union)
    else:
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
    return iou_score.item()


def normalize(name, feature, norms):
    if name.startswith('E'):
        max_val = np.max(norms['E'])
    elif name.startswith('B'):
        max_val = np.max(norms['B'])
    elif name.startswith('v'):
        max_val = np.max(norms['v'])
    else:
        max_val = np.max(feature)
    return feature / max_val


def standardize(feature):
    avg = np.mean(feature)
    std = np.std(feature)
    return (feature - avg) / std


def euclidian(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True

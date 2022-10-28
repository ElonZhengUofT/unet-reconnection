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

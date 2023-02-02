import torch
import numpy as np


def iou_score(prediction, target):
    if torch.is_tensor(target) and torch.is_tensor(prediction):
        intersection = torch.logical_and(target, prediction)
        union = torch.logical_or(target, prediction)
        iou_score = torch.sum(intersection) / torch.sum(union).item()
    else:
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def f_beta(precision, recall, beta):
    numerator = (1 + beta**2) * precision * recall
    denominator = beta**2 * precision + recall
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0))


def pick_best_threshold(precision, recall, thresholds, beta):
    f_scores = f_beta(precision, recall, beta)
    max_f_score = np.max(f_scores)
    max_f_index = np.argmax(f_scores)
    max_f_thresh = thresholds[max_f_index]
    return max_f_score, max_f_index, max_f_thresh


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


def split_data(files, file_fraction, data_splits):
    num_files = file_fraction * len(files)
    train_split, val_split, test_split = data_splits
    train_index = int(train_split * num_files)
    val_index = train_index + int(val_split * num_files)
    test_index = val_index + int(test_split * num_files)
    train_files = files[:train_index]
    val_files = files[train_index:val_index]
    test_files = files[val_index:test_index]
    return train_files, val_files, test_files

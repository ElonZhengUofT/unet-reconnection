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

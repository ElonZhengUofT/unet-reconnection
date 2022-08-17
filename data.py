import torch
from torch.utils.data import Dataset
import numpy as np


class NpzDataset(Dataset):
    def __init__(self, npz_file_paths, features):
        self.files = npz_file_paths
        self.features = features
    
    def __getitem__(self, index):
        data = np.load(self.files[index])
        X = np.stack([data[feature] for feature in self.features], axis=0)
        y = data['labeled_domain']
        return {
            'X': torch.tensor(X, dtype=torch.float32), 
            'y': torch.tensor(y, dtype=torch.float32)
        }
    
    def __len__(self):
        return len(self.files)
        
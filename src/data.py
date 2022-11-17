import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils import normalize, standardize, euclidian
from pathlib import Path


class NpzDataset(Dataset):
    def __init__(self, npz_file_paths, features, normalize, standardize, binary):
        self.files = npz_file_paths
        self.features = features
        self.normalize = normalize
        self.standardize = standardize
        self.binary = binary
    
    def __getitem__(self, index):
        data = np.load(self.files[index])

        earth = data['rho'] == 0
        not_earth = ~earth

        if self.normalize:
            abs_E = euclidian(data['Ex'], data['Ey'], data['Ez'])
            abs_B = euclidian(data['Bx'], data['By'], data['Bz'])
            abs_v = euclidian(data['vx'], data['vy'], data['vz'])
            norms = {
                'E': abs_E[not_earth], 
                'B': abs_B[not_earth], 
                'v': abs_v[not_earth],
            }

        data_dict = {}
        for feature_name in self.features:
            current_feature = data[feature_name].copy()
            
            if self.normalize:
                current_feature[not_earth] = normalize(feature_name, current_feature[not_earth], norms)
            elif self.standardize:
                current_feature[not_earth] = standardize(current_feature[not_earth])

            current_feature[earth] = 0
            data_dict[feature_name] = current_feature
        
        X = np.stack([data_dict[feature_name] for feature_name in self.features], axis=0)
        
        if self.binary:
            y = data['labeled_domain'][np.newaxis,:,:]
            not_earth = not_earth[np.newaxis,:,:]
        else:
            label = data['labeled_domain']
            opposite_label = np.where(label, 1, 0)
            y = np.stack((label, opposite_label))
            not_earth = np.stack((not_earth, not_earth))

        return {
            'X': torch.tensor(X, dtype=torch.float32), 
            'y': torch.tensor(y, dtype=torch.float32),
            'not_earth': torch.tensor(not_earth, dtype=torch.bool),
            'fname': Path(self.files[index]).stem
        }
    
    def __len__(self):
        return len(self.files)
        
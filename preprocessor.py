import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import numpy as np

# returns X and y from either the train or test sets depending on train 
# bool value

# returns X and y for either train or test
def load_data(train = bool) -> tuple[np.ndarray , np.ndarray]:
    path = Path.cwd()
    
    # get filepaths for x and y data and load the y data
    if train == True:
        X_dir = path / 'UCI HAR Dataset/train/Inertial Signals'
        y_dir = path / 'UCI HAR Dataset/train'
        y = np.loadtxt(y_dir / 'y_train.txt')
        X = np.empty(shape=(7352,1,128))
    elif train == False:
        X_dir = path / 'UCI HAR Dataset/test/Inertial Signals'
        y_dir = path / 'UCI HAR Dataset/test'
        y = np.loadtxt(y_dir / 'y_test.txt')
        X = np.empty(shape=(2947,1,128))
    else: 
        raise NameError("train value must be boolean")
        
    # concatenate all individual channel files into one X array.
    channel_files = os.listdir(X_dir)[:6] # omitting total_acc channels
    for file in channel_files:
        x_channel = np.loadtxt(X_dir / file)
        x_channel = np.expand_dims(x_channel, axis=1)
        X = np.concatenate((X, x_channel), axis=1)
    X = np.delete(arr=X, obj=0, axis=1)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).long() - 1
    
    return X, y


class create_dataset(Dataset):
    """load human activity recognition dataset"""
    
    def __init__(self, train: bool, transform=None, 
                 target_transform=None):
        self.X, self.y = load_data(train)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx, :, :]
        label = self.y[idx]
        return [sample, label]


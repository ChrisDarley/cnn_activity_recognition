import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

# achieves max 87% accuracy
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=6)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=13),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=13),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop = nn.Dropout(p=0.6)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*23, out_features=500),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=500, out_features=50),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=50, out_features=6))
        

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.reshape(64, -1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


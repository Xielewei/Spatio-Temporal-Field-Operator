"""Adapt cached stream windows to the common forecasting interface."""

import torch
from torch_geometric.data import Data, Dataset


class SpatioTemporalDataset(Dataset):
    def __init__(self, inputs, split):
        self.x = inputs[split + "_x"]
        self.y = inputs[split + "_y"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y)

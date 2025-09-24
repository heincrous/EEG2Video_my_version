import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


# ================================
# Helper to dynamically compute flatten size
# ================================
def _get_flattened_size(net, C, T):
    with torch.no_grad():
        x = torch.zeros(1, 1, C, T)
        y = net(x)
        return y.view(1, -1).shape[1]


# ================================
# ShallowNet
# ================================
class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


# ================================
# DeepNet
# ================================
class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


# ================================
# EEGNet
# ================================
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


# ================================
# TsConv
# ================================
class tsconv(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

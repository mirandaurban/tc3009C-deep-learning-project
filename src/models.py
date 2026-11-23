"""Regression estimators used in the experiments."""
import torch.nn as nn
from sklearn.linear_model import Ridge

def make_ridge(alpha=1.0):
    return Ridge(alpha=alpha)

class MLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers, n = [], n_in
        for width in hidden:
            layers.extend((nn.Linear(n, width), nn.ReLU(), nn.Dropout(dropout)))
            n = width
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

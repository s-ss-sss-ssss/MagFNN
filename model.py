"""Small direct field/gradient regressor; no implicit position prediction."""
import torch
from torch import nn


class DipoleNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, width),
                                    nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 8))

    def forward(self, x):
        return self.layers(x)


def loss_parts(prediction, target):
    field = (prediction[:, :3]-target[:, :3]).square().mean()
    gradient = (prediction[:, 3:]-target[:, 3:]).square().mean()
    return field+gradient, field, gradient


def select_device(name):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available; select --device cpu')
    return torch.device(name)

import torch
from torch import nn. einsum
import numpy as np
from einops import rearrange, repeat



class CyclicShift(nn.Moduel):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts = (self.displacement, self.displacement), dims = (1, 2))


class Residuals(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
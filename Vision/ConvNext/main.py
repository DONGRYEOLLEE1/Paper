import torch
import torch.nn as nn
from typing import List


class ConvNormAct(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, norm = nn.BatchNorm2d, act = nn.ReLU, **kwargs):
        super(ConvNormAct, self).__init__()
        
        self.
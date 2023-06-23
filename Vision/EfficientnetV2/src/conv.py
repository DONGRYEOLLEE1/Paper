import torch
import torch.nn as nn

from .misc import SqueezeExcitation, ConvBnAct
from .stochastic_depth import StochasticDepth

class MBConvN(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 expansion_factor = 4,
                 reduction_factor = 4,
                 survival_prob = .8
                 ):
        super(MBConvN, self).__init__()
        
        reduced_dim = int(in_channels // 4)
        expanded_dim = int(expansion_factor * in_channels)
        padding = (kernel_size - 1) // 2
        
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(in_channels, expanded_dim, kernel_size = 1)
        self.depthwise_conv = ConvBnAct(expanded_dim, expanded_dim,
                                        kernel_size, stride = stride,
                                        padding = padding, groups = expanded_dim)
        self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = ConvBnAct(expanded_dim, out_channels, kernel_size = 1, act = False)
        
    def forward(self, x):
        
        residual = x.clone()
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            x += residual
            
        return x
    

class FusedMBConvN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 expansion_factor = 4,
                 reduction_factor = 4,
                 survival_prob = .8
                 ):
        super(FusedMBConvN, self).__init__()
        
        reduced_dim = int(in_channels // 4)
        expanded_dim = int(expansion_factor * in_channels)
        padding = (kernel_size - 1)//2
        
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        self.conv = ConvBnAct(in_channels, expanded_dim,
                              kernel_size, stride = stride,
                              padding = padding, groups = 1
                             )
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = nn.Identity() if (expansion_factor == 1) else ConvBnAct(expanded_dim, out_channels, kernel_size = 1, act = False)
        
    def forward(self, x):
        
        residual = x.clone()
        x = self.conv(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            x += residual
            
        return x
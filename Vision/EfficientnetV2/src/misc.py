import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Conv2d(in_channels, reduced_dim, kernel_size = 1),
                                    nn.SiLU(),
                                    nn.Conv2d(reduced_dim, in_channels, kernel_size = 1),
                                    nn.Sigmoid()
                                    )
        
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        
        return x * y
    
class ConvBnAct(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 0,
                 groups = 1,
                 act = True,
                 bn = True,
                 bias = False
                 ):
        super(ConvBnAct, self).__init__()
        
        self.conv = nn.Conv2d(in_channels,
                              out_channels, 
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              groups = groups,
                              bias = bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
import torch
import torch.nn as nn

from .conv import MBConvN, FusedMBConvN
from .misc import ConvBnAct


class EfficientNetV2(nn.Module):
    def __init__(self, 
                 v2_set,
                 version = "s",
                 dropout_rate = .2,
                 num_classes = 1000
                 ):
        super(EfficientNetV2, self).__init__()
        
        last_channel = 1280
        self.features = self._feature_extractor(v2_set, version, last_channel)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace = True),
            nn.Linear(last_channel, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x
    
    def _feature_extractor(self, v2_set, version, last_channel):
        
        # Extract the Config
        config = v2_set[version]
        
        layers = []
        layers.append(ConvBnAct(3, config[0][3], kernel_size = 3, stride = 2, padding = 1))
        
        for (expansion_factor, k, stride, in_channels, out_channels, num_layers, use_fused) in config:
            
            if use_fused:
                layers += [FusedMBConvN(in_channels if repeat == 0 else out_channels,
                                        out_channels,
                                        kernel_size = k,
                                        stride = stride if repeat == 0 else 1,
                                        expansion_factor = expansion_factor) for repeat in range(num_layers)
                           ]
            else:
                layers += [MBConvN(in_channels if repeat == 0 else out_channels,
                                   out_channels,
                                   kernel_size = k,
                                   stride = stride if repeat == 0 else 1,
                                   expansion_factor = expansion_factor) for repeat in range(num_layers)
                           ]
                
        layers.append(ConvBnAct(config[-1][4], last_channel, kernel_size = 1))
        
        return nn.Sequential(*layers)
import torch
import torch.nn as nn

from .conv import MBConvN, FusedMBConvN
from .misc import ConvBnAct

V2_set = {
    # expansion_factor, k, stride, n_in, n_out, num_layers, use_fusedMBCONV
    "s" : [
        [1, 3, 1, 24, 24, 2, True],
        [4, 3, 2, 24, 48, 4, True],
        [4, 3, 2, 48, 64, 4, True],
        [4, 3, 2, 64, 128, 6, False],
        [6, 3, 1, 128, 160, 9, False],
        [6, 3, 2, 160, 256, 15, False]
    ],
    "m" : [
        [1, 3, 1, 24, 24, 3, True],
        [4, 3, 2, 24, 48, 5, True],
        [4, 3, 2, 48, 80, 5, True],
        [4, 3, 2, 80, 160, 7, False],
        [6, 3, 1, 160, 176, 14, False],
        [6, 3, 2, 176, 304, 18, False],
        [6, 3, 1, 304, 512, 5, False]
    ],
    "l" : [
        [1, 3, 1, 32, 32, 4, True],
        [4, 3, 2, 32, 64, 7, True],
        [4, 3, 2, 64, 96, 7, True],
        [4, 3, 2, 96, 192, 10, False],
        [6, 3, 1, 192, 224, 19, False],
        [6, 3, 2, 224, 384, 25, False],
        [6, 3, 1, 384, 640, 7, False]
    ]
}


class EfficientNetV2(nn.Module):
    def __init__(self,
                 version = "s",
                 dropout_rate = .2,
                 num_classes = 1000
                 ):
        super(EfficientNetV2, self).__init__()
        
        last_channel = 1280
        self.features = self._feature_extractor(version, last_channel)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace = True),
            nn.Linear(last_channel, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x
    
    def _feature_extractor(self, version, last_channel):
        
        # Extract the Config
        config = V2_set[version]
        
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
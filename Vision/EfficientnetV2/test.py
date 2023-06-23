import torch
import torch.nn as nn

from src.efficientv2 import EfficientNetV2

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

def test(version = 's', num_classes = 1000):
    net = EfficientNetV2(v2_set= V2_set, version = version, num_classes = num_classes)
    x = torch.rand(4, 3, 224, 224)
    output = net(x)
    
    print(output.size())
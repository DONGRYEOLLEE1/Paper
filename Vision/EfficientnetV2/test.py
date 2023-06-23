import torch
import torch.nn as nn

from src.efficientv2 import EfficientNetV2


def test(version = 'm', num_classes = 1000):
    net = EfficientNetV2(version = version, num_classes = num_classes)
    x = torch.rand(4, 3, 224, 224)
    output = net(x)
    
    print(output.size())
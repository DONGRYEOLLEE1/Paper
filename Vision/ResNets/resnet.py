import torch
import torch.nn as nn
import torch.nn.functional as F


## Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()


        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Identity Shortcut
        self.shortcut = nn.Sequential()

        # size가 안맞아서 합연산이 불가하다면 연산 가능한 형태로 맞춰주기
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_planes)
            )

        
    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.bn1(x_)
        x_ = F.relu(x_)
        
        x_ = self.conv2(x_)
        x_ = self.bn2(x_)
        x_ += self.shortcut(x)
        x_ = F.relu(x_)

        return x_


class BottleNeck(nn.Module):

    mul = 4

    def __init__(self, in_planes, out_planes, stride = 1):
        super(BottleNeck, self).__init__()

        # 첫 Conv는 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    
    def forward(self, x):

        x_ = self.conv1(x)
        x_ = self.bn1(x_)
        x_ = F.relu(x_)

        x_ = self.conv2(x_)
        x_ = self.bn2(x_)
        x_ = F.relu(x_)

        x_ = self.conv3(x_)
        x_ = self.bn3(x_)
        x_ += self.shortcut(x)
        x_ = F.relu(x_)

        return x_


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes = 10):
        super(ResNet, self).__init__()
        # RGB 3개 채널에서 64개의 kernel 사용
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, num_block[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.mul, num_classes)



    def _make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2, ])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


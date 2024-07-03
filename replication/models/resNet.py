import numpy as np
import torch
import torch.nn as nn
# import torchvision.models
from torchvision.models import resnet18, ResNet18_Weights


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def get_features(loader):
    # resnet = resnet18(weights=ResNet18_Weights.DEFAULT, pretrained=True)
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.cuda()
    resnet.eval()

    features_acc = []
    for i, batch in enumerate(loader):
        input, _ = batch
        input = input.cuda()
        x = resnet.conv1(input)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)
        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)
        # features = resnet(input).detach()
        features = resnet.avgpool(x).detach().squeeze()
        features_acc.append(features)
    res= torch.cat(features_acc)
    return res

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        # KaiMing He's initialization (Done on paper)
        def _weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        self.apply(_weights_init)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            # downsample = nn.Sequential(
            #     conv3x3(self.in_channels, out_channels, stride=stride),
            #     nn.BatchNorm2d(out_channels))
            # LambdaLayer used by CIFAR10 ResNet paper
            downsample = LambdaLayer(lambda x:
                                        nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
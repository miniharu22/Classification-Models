import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        return x
    

class Inception(nn.Module):
    def __init__(self, in_channels, n1, n3_reduce, n3, n5_reduce, n5, pool_proj):
        super().__init__()
        self.branch1 = ConvBlock(in_channels, n1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
          ConvBlock(in_channels, n3_reduce, kernel_size=1, stride=1, padding=0),
          ConvBlock(n3_reduce, n3, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
          ConvBlock(in_channels, n5_reduce, kernel_size=1, stride=1, padding=0),
          ConvBlock(n5_reduce, n5, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
          ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], dim=1)
    

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avg_conv = nn.Sequential(
          nn.AvgPool2d(kernel_size=5, stride=3),
          ConvBlock(in_channels, 128, kernel_size=1, stride=1, padding=0)
        )

        self.fc = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=0.7),
          nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.avg_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=10):
        super().__init__()

        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool3(x)
        x = self.a4(x)
        
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.dropout(x)

        if self.aux_logits and self.training:
            return [x, aux1, aux2]
        else:
            return x
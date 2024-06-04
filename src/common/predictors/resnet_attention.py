import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        conv_layer1 = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        conv_layer2 = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer1,
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer2,
        )

    def forward(self, x):
        return x + self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self, num_hidden):
        super(AttentionBlock, self).__init__()

        self.num_hidden = num_hidden

    def forward(self, x):
        return x


class ResNetAttention(nn.Module):
    def __init__(self, num_hidden):
        super(ResNetAttention, self).__init__()

        self.num_hidden = num_hidden

        self.inconv = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))

        self.attention = AttentionBlock(num_hidden)
        self.backBone  = nn.ModuleList([ResBlock(num_hidden)] * 4)

        self.outconv = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.inconv(x)
        x = self.attention(x)

        for block in self.backBone:
            x = block(x)

        x = self.outconv(x)
        return self.sigmoid(x)


    def name(self):
        str_network = "ResNetAttention_"
        str_network += f"{self.num_hidden}hidden"
        return str_network


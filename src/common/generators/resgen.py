import torch
import torch.nn as nn

from configs.constants import *


class ResGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.startBlock = nn.Conv2d(GRID_NUM_CHANNELS, 32, kernel_size=3, stride=1, padding=1)

        self.backBone = nn.ModuleList(
            [ResBlock(32) for _ in range(4)]
        )

        self.out_conv = nn.Conv2d(32, GRID_NUM_CHANNELS, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        x = self.out_conv(x)

        x = torch.tanh(x)

        return x

    def name(self):
        str_network = "ResGen"

        return str_network


class ResBlock(nn.Module):
    def __init__(self, num_hidden, topology):
        super().__init__()

        conv_layer = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer,
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer,
        )

    def forward(self, x):
        return x + self.block(x)


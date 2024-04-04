import torch
import torch.nn as nn

from src.common.predictors.utils.toroidal import toroidal_Conv2d
from configs.constants import *


class block(nn.Module):
    def __init__(self, channels:int, block_dim:int=2) -> None:
        super(block, self).__init__()

        self.channels  = channels
        self.block_dim = block_dim

        self.convs = nn.ModuleList([nn.Conv2d(self.channels, self.channels,
                                              kernel_size = 3, stride = 1, padding = 0) for _ in range(self.block_dim)])
        self.bn    = nn.BatchNorm2d(self.channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        identity = x

        for conv in self.convs:
            x = self.bn(x)
            x = self.relu(x)
            x = toroidal_Conv2d(x, conv, padding = 1)

        x += identity

        return x


class ResNet(nn.Module):
    """
    ResNet model

    Each layer consists of two blocks

    """
    def __init__(self, n_layers, channels):
        super(ResNet, self).__init__()
        self.channels = channels
        self.n_layers = n_layers

        self.in_conv  = nn.Conv2d(N_CHANNELS, self.channels, kernel_size=3, padding=0)
        self.out_conv = nn.Conv2d(self.channels, N_CHANNELS, kernel_size=1, padding=0)

        self.layers   = self._make_layer(block, self.n_layers, self.channels)


    def forward(self, x):

        x = toroidal_Conv2d(x, self.in_conv)

        for layer in self.layers:
            x = layer(x)

        x = self.out_conv(x)

        return x


    def _make_layer(self, block, n_layers, channels):
        layers = []

        for _ in range(n_layers):
            layers.append(block(channels))

        return nn.Sequential(*layers)


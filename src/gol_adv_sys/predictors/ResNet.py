import torch
import torch.nn as nn

from ..utils.helper_functions import add_toroidal_padding
from ..utils import constants as constants


class block(nn.Module):
    def __init__(self, channels) -> None:
        super(block, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 0)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.bn(x)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv)

        x = self.bn(x)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv)

        x += identity

        return x

    def _pad_conv(self, x, padding=1, f=None):
        x = add_toroidal_padding(x, padding)
        x = f(x)

        return x


class ResNetConstantChannels(nn.Module):
    def __init__(self, layers, channels):
        super(ResNetConstantChannels, self).__init__()
        self.channels = channels
        self.n_layers = len(layers)

        self.in_conv = nn.Conv2d(constants.nc, channels, kernel_size=3, padding=0)
        self.out_conv = nn.Conv2d(channels, constants.nc, kernel_size=1, padding=0)

        for i in range(self.n_layers):
            setattr(self, f"layer{i}", self._make_layer(block, layers[i], self.channels))


    def forward(self, x):

        x = self._pad_conv(x, self.in_conv)

        for i in range(self.n_layers):
            x = getattr(self, f"layer{i}")(x)

        x = self.out_conv(x)

        return x


    def _pad_conv(self, x, padding=1, f=None):
        x = add_toroidal_padding(x, padding)
        x = f(x)

        return x


    def _make_layer(self, block, num_residual_blocks, channels):
        layers = []

        for _ in range(num_residual_blocks):
            layers.append(block(channels))
            layers.append(block(channels))

        return nn.Sequential(*layers)


import torch
import torch.nn as nn

from .utils import constants as constants

class block(nn.Module):
    def __init__(self, channels) -> None:
        super(block, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        idetity = x

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        x += idetity

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.channels = constants.npf
        self.conv1 = nn.Conv2d(constants.nc, self.channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

        self.n_layers = len(layers)

        for i in range(self.n_layers):
            setattr(self, f"layer{i}", self._make_layer(block, layers[i], self.channels))

        self.conv2 = nn.Conv2d(self.channels, constants.nc, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i in range(self.n_layers):
            x = getattr(self, f"layer{i}")(x)

        x = self.conv2(x)

        return x

    def _make_layer(self, block, num_residual_blocks, channels):

        layers = []

        for _ in range(num_residual_blocks):
            layers.append(block(channels))
            layers.append(block(channels))

        return nn.Sequential(*layers)


def Predictor_34():
    return ResNet(block, [3, 4, 6, 3])


def Predictor_18():
    return ResNet(block, [2, 2, 2, 2])



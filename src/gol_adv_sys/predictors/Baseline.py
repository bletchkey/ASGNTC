import torch
import torch.nn as nn

from ..utils.helper_functions import add_toroidal_padding
from ..utils import constants as constants


class Baseline(nn.Module):

    def __init__(self) -> None:
        super(Baseline, self).__init__()

        # First block
        self.conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(constants.npf, constants.npf, kernel_size=3, padding=0)

        # Second block
        self.conv3 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(constants.npf*2, constants.npf*2, kernel_size=3, padding=0)

        # Third block
        self.conv5 = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(constants.npf*4, constants.npf*4, kernel_size=3, padding=0)

        # Output Convolution
        self.output_conv = nn.Conv2d(constants.npf*4, constants.nc, kernel_size=1, padding=0)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self._pad_conv(x, self.conv1)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv2)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv3)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv4)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv5)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv6)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv6)
        x = self.relu(x)

        x = self.output_conv(x)

        return x

    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f(x)

        return x


class Baseline_v2(nn.Module):
    def __init__(self) -> None:
        super(Baseline_v2, self).__init__()

        self.conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(constants.npf*4, constants.npf, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(constants.npf*2, constants.npf*2, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(constants.npf*4, constants.npf*8, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(constants.npf*8, constants.npf*8, kernel_size=3, padding=0)
        self.output_conv = nn.Conv2d(constants.npf*8, constants.nc, kernel_size=1, padding=0)


    def forward(self, x):

        x1 = self._pad_conv(x, 1, self.conv1)
        x1 = torch.relu(x1)

        x2 = self._pad_conv(x, 1, self.conv1)
        x2 = torch.relu(x2)

        x3 = self._pad_conv(x, 1, self.conv1)
        x3 = torch.relu(x3)

        x4 = self._pad_conv(x, 1, self.conv1)
        x4 = torch.relu(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self._pad_conv(x, 1, self.conv2)
        x = torch.relu(x)

        x = self._pad_conv(x, 1, self.conv3)
        x = torch.relu(x)

        x1 = self._pad_conv(x, 1, self.conv4)
        x1 = torch.relu(x1)

        x2 = self._pad_conv(x, 1, self.conv4)
        x2 = torch.relu(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self._pad_conv(x, 1, self.conv5)
        x = torch.relu(x)

        x = self._pad_conv(x, 1, self.conv6)
        x = torch.relu(x)

        x = self.output_conv(x)


        return x


    def _pad_conv(self, x, padding, f):
        for _ in range(padding):
            x = add_toroidal_padding(x)
        x = f(x)

        return x


class Baseline_v3(nn.Module):

    def __init__(self) -> None:
        super(Baseline_v3, self).__init__()

        # First step
        self.in_conv = nn.Conv2d(constants.nc, 96, kernel_size=3, padding=0)

        # Second step
        self.conv_1x1 = nn.Conv2d(16, 256, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(16, 128, kernel_size=3, padding=0)
        self.conv_5x5 = nn.Conv2d(16, 64, kernel_size=5, padding=0)
        self.conv_7x7 = nn.Conv2d(16, 32, kernel_size=7, padding=0)

        # Third step
        self.out_conv = nn.Conv2d(512, constants.nc, kernel_size=1, padding=0)

        #Activation Function
        self.relu = nn.ReLU()


    def forward(self, x):

        for _ in range(2):
            x = self._pad_conv(x, 1, self.in_conv)
            x = self.relu(x)

            parts = [x[:, i:i+16, :, :] for i in range(0, 64, 16)]
            parts.append(x[:, 64:, :, :])

            x_1x1 = self.conv_1x1(parts[0])
            x_1x1 = self.relu(x_1x1)

            x_3x3 = self._pad_conv(parts[1], 1, self.conv_3x3)
            x_3x3 = self.relu(x_3x3)

            x_5x5 = self._pad_conv(parts[2], 2, self.conv_5x5)
            x_5x5 = self.relu(x_5x5)

            x_7x7 = self._pad_conv(parts[3], 3, self.conv_7x7)
            x_7x7 = self.relu(x_7x7)

            x = torch.cat([x_1x1, x_3x3, x_5x5, x_7x7, parts[4]], dim=1)

            x = self.out_conv(x)
            x = self.relu(x)

        return x


    def _pad_conv(self, x, padding, f):
        for _ in range(padding):
            x = add_toroidal_padding(x)
        x = f(x)

        return x



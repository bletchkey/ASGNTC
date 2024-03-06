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
        self.output_conv = nn.Conv2d(constants.npf*2, constants.nc, kernel_size=1, padding=0)

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

        # x = self._pad_conv(x, self.conv5)
        # x = self.relu(x)
        # x = self._pad_conv(x, self.conv6)
        # x = self.relu(x)
        # x = self._pad_conv(x, self.conv6)
        # x = self.relu(x)

        x = self.output_conv(x)

        return x

    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f(x)

        return x


# Test model
class VGGLike_13(nn.Module):

    def __init__(self) -> None:
        super(VGGLike_13, self).__init__()

        # First block
        self.conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(constants.npf, constants.npf, kernel_size=3, padding=0)

        # Second block
        self.conv3 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(constants.npf*2, constants.npf*2, kernel_size=3, padding=0)

        # Third block
        self.conv5 = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(constants.npf*4, constants.npf*4, kernel_size=3, padding=0)

        # Fourth block
        self.conv7 = nn.Conv2d(constants.npf*4, constants.npf*8, kernel_size=3, padding=0)
        self.conv8 = nn.Conv2d(constants.npf*8, constants.npf*8, kernel_size=3, padding=0)

        # Output Convolution
        self.output_conv = nn.Conv2d(constants.npf*8, constants.nc, kernel_size=1, padding=0)

        # Activation Function
        self.relu = nn.ReLU()

        # Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

    def forward(self, x):

        x = self._pad_conv(x, self.conv1)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv2)
        x = self.relu(x)

        x = add_toroidal_padding(x)
        x = self.avgpool(x)

        x = self._pad_conv(x, self.conv3)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv4)
        x = self.relu(x)

        x = add_toroidal_padding(x)
        x = self.avgpool(x)

        x = self._pad_conv(x, self.conv5)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv6)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv6)
        x = self.relu(x)

        x = add_toroidal_padding(x)
        x = self.avgpool(x)

        x = self._pad_conv(x, self.conv7)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv8)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv8)
        x = self.relu(x)

        x = add_toroidal_padding(x)
        x = self.avgpool(x)

        x = self._pad_conv(x, self.conv8)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv8)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv8)
        x = self.relu(x)

        x = add_toroidal_padding(x)
        x = self.avgpool(x)

        x = self.output_conv(x)

        return x

    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f(x)

        return x


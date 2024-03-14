import torch
import torch.nn as nn

from .utils.toroidal import toroidal_Conv2d
from ..utils import constants as constants


class GloNet(nn.Module):

    def __init__(self):
        super(GloNet, self).__init__()

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
        self.output_conv = nn.Conv2d(constants.npf, constants.nc, kernel_size=1, padding=0)

        # Adjusting Convolutions
        self.adjust_conv1 = nn.Conv2d(in_channels=constants.npf, out_channels=constants.npf, kernel_size=1)
        self.adjust_conv2 = nn.Conv2d(in_channels=constants.npf*2, out_channels=constants.npf, kernel_size=1)
        self.adjust_conv3 = nn.Conv2d(in_channels=constants.npf*4, out_channels=constants.npf, kernel_size=1)


        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = toroidal_Conv2d(x, self.conv1)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv2)
        x = self.relu(x)

        x1 = x

        x = toroidal_Conv2d(x, self.conv3)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv4)
        x = self.relu(x)

        x2 = x

        x = toroidal_Conv2d(x, self.conv5)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv6)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv6)
        x = self.relu(x)

        x3 = x

        x1 = self.adjust_conv1(x1)
        x2 = self.adjust_conv2(x2)
        x3 = self.adjust_conv3(x3)

        x = x1 + x2 + x3

        x = self.output_conv(x)

        return x


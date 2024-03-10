import torch
import torch.nn as nn

from ..utils.helper_functions import add_toroidal_padding
from ..utils import constants as constants

class Conv2dToroidalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dToroidalLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.padding = padding

    def forward(self, x):
        for _ in range(self.padding):
            x = add_toroidal_padding(x)

        x = self.conv(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, n7x7red, n7x7):
        super(InceptionModule, self).__init__()

        self.conv_3x3 = Conv2dToroidalLayer(n3x3red, n3x3, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = Conv2dToroidalLayer(n5x5red, n5x5, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = Conv2dToroidalLayer(n7x7red, n7x7, kernel_size=7, stride=1, padding=3)

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            self.conv_3x3,
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            self.conv_5x5,
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 1x1 conv -> 7x7 conv branch
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, n7x7red, kernel_size=1),
            nn.BatchNorm2d(n7x7red),
            nn.ReLU(True),
            self.conv_7x7,
            nn.BatchNorm2d(n7x7),
            nn.ReLU(True),
        )

    def forward(self, x):
        return torch.cat([
            self.b1(x),
            self.b2(x),
            self.b3(x),
            self.b4(x)
        ], 1)


class GoogLeNet(nn.Module):
    def __init__(self):

        super(GoogLeNet, self).__init__()

        self.pre_conv = Conv2dToroidalLayer(constants.nc, 64, kernel_size=3, stride=1, padding=1)

        self.pre_layers = nn.Sequential(
            self.pre_conv,
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.inc_module  = InceptionModule(in_channels=64, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32, n7x7red=16, n7x7=32)
        self.inc_module2 = InceptionModule(in_channels=256, n1x1=128, n3x3red=128, n3x3=192, n5x5red=32, n5x5=96, n7x7red=32, n7x7=96)
        self.output_conv = nn.Conv2d(512, constants.nc, kernel_size=1, padding=0)


    def forward(self, x):

        x = self.pre_layers(x)
        x = self.inc_module(x)
        x = self.inc_module2(x)
        x = self.output_conv(x)

        return x


import torch
import torch.nn as nn

from ..utils.helper_functions import add_toroidal_padding
from ..utils import constants as constants


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.enc_conv2 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)

        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)

        # Decoder (Upsampling)
        self.up_conv1 = nn.ConvTranspose2d(constants.npf*4, constants.npf*2, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(constants.npf*4, constants.npf*2, kernel_size=3, padding=0)
        self.up_conv2 = nn.ConvTranspose2d(constants.npf*2, constants.npf, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(constants.npf*2, constants.npf, kernel_size=3, padding=0)

        # Output Convolution
        self.output_conv = nn.Conv2d(constants.npf, constants.nc, kernel_size=1, padding = 0)

        # Activation Function
        self.relu = nn.ReLU()


    def forward(self, x):

        for _ in range(1):
            x = self._u_structure(x)

        return x


    def _pad_conv(self, x, padding=1, f=None):
        x = add_toroidal_padding(x, padding)
        x = f(x)

        return x

    def _u_structure(self, x):

        # Encoder
        x = self.relu(x)

        enc_1 = self._pad_conv(x, self.enc_conv1)
        x = self.avgpool(enc_1)
        x = self.relu(x)

        enc_2 = self._pad_conv(x, self.enc_conv2)
        x = self.avgpool(enc_2)
        x = self.relu(x)

        # Bottleneck
        x = self._pad_conv(x, self.bottleneck_conv)

        # Decoder
        dec_1 = self.up_conv1(x)
        x = torch.cat((dec_1, enc_2), dim=1)
        x = self.relu(x)
        x = self._pad_conv(x, self.dec_conv1)

        dec_2 = self.up_conv2(x)
        x = torch.cat((dec_2, enc_1), dim=1)
        x = self.relu(x)
        x = self._pad_conv(x, self.dec_conv2)

        # Output
        out = self.output_conv(x)

        return out


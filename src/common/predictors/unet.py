import torch
import torch.nn as nn

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = ToroidalConv2d(nn.Conv2d(GRID_NUM_CHANNELS, N_PREDICTOR_FEATURES, kernel_size=3, padding=0))
        self.enc_conv2 = ToroidalConv2d(nn.Conv2d(N_PREDICTOR_FEATURES, N_PREDICTOR_FEATURES*2, kernel_size=3, padding=0))

        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = ToroidalConv2d(nn.Conv2d(N_PREDICTOR_FEATURES*2, N_PREDICTOR_FEATURES*4, kernel_size=3, padding=0))

        # Decoder (Upsampling)
        self.up_conv1  = nn.ConvTranspose2d(N_PREDICTOR_FEATURES*4, N_PREDICTOR_FEATURES*2, kernel_size=2, stride=2)
        self.dec_conv1 = ToroidalConv2d(nn.Conv2d(N_PREDICTOR_FEATURES*4, N_PREDICTOR_FEATURES*2, kernel_size=3, padding=0))
        self.up_conv2  = nn.ConvTranspose2d(N_PREDICTOR_FEATURES*2, N_PREDICTOR_FEATURES, kernel_size=2, stride=2)
        self.dec_conv2 = ToroidalConv2d(nn.Conv2d(N_PREDICTOR_FEATURES*2, N_PREDICTOR_FEATURES, kernel_size=3, padding=0))

        # Output Convolution
        self.output_conv = nn.Conv2d(N_PREDICTOR_FEATURES, GRID_NUM_CHANNELS, kernel_size=1, padding = 0)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self._u_structure(x)

        x = torch.sigmoid(x)

        return x

    def name(self):
        str_network = "UNet"
        return str_network

    def _u_structure(self, x):

        # Encoder
        x = self.relu(x)

        enc_1 = self.enc_conv1(x)
        x = self.avgpool(enc_1)
        x = self.relu(x)

        enc_2 = self.enc_conv2(x)
        x = self.avgpool(enc_2)
        x = self.relu(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        dec_1 = self.up_conv1(x)
        x = torch.cat((dec_1, enc_2), dim=1)
        x = self.relu(x)
        x = self.dec_conv1(x)

        dec_2 = self.up_conv2(x)
        x = torch.cat((dec_2, enc_1), dim=1)
        x = self.relu(x)
        x = self.dec_conv2(x)

        # Output
        out = self.output_conv(x)

        return out


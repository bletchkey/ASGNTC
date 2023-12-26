import torch
import torch.nn as nn

from .utils import constants as constants


class GeneratorDC(nn.Module):
    def __init__(self):
        super(GeneratorDC, self).__init__()

        layers = [
            self._make_layer(constants.nz, constants.ngf * 8, kernel_size=4, stride=1, padding=0),
            self._make_layer(constants.ngf * 8, constants.ngf * 4),
            self._make_layer(constants.ngf * 4, constants.ngf * 2),
            self._make_final_layer(constants.ngf * 2, constants.nc)
        ]

        self.model = nn.Sequential(*layers)


    def _make_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def _make_final_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=True),
            nn.Tanh()
        )


    def forward(self, input):
        return self.model(input)


def Generator():
    return GeneratorDC()


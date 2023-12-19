import torch
import torch.nn as nn

from .utils import constants as constants


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(constants.nz, constants.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(constants.ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(constants.ngf * 8, constants.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(constants.ngf * 4, constants.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(constants.ngf * 2, constants.nc, 4, 2, 1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)


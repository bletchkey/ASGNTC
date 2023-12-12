import torch
import torch.nn as nn

from . import constants as constants

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(constants.nz, constants.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(constants.ngf * 8),
            nn.ReLU(True),


            nn.ConvTranspose2d(constants.ngf * 8, constants.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(constants.ngf * 4, constants.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(constants.ngf * 2, constants.nc, 4, 2, 1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(constants.nc,constants.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(constants.ndf * 2, constants.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(constants.ndf * 4, constants.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(constants.ngf * 8, constants.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(constants.ngf * 4, constants.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(constants.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(constants.ngf * 2, constants.nc, 4, 2, 1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)
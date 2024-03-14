import torch
import torch.nn as nn

from ..utils import constants as constants

class DCGAN(nn.Module):
    def __init__(self, noise_std=0) -> None:
        super(DCGAN, self).__init__()

        self.noise_std = noise_std

        layers = [
            self._make_layer(constants.nz, constants.ngf * 4, kernel_size=4, stride=1, padding=0),
            self._make_layer(constants.ngf * 4, constants.ngf * 2),
            self._make_layer(constants.ngf * 2, constants.ngf),
            self._make_final_layer(constants.ngf, constants.nc),
        ]

        self.model = nn.Sequential(*layers)
        self.alpha = 1000

    def _make_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_final_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=True),
        )

    def threshold(self, x, alpha):
        return (1 + nn.Tanh()(alpha * x)) / 2

    def forward(self, input):
        x = self.model(input)
        # Add variance to the output
        if self.noise_std > 0:
            # mean_shift = random.uniform(-8, 8)
            mean_shift = random.normalvariate(0, 6)
            noise = (torch.randn_like(x) * self.noise_std) + mean_shift
            x = x + noise
        x = self.threshold(x, self.alpha)
        return x


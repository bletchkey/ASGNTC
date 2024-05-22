import torch
import torch.nn as nn
import random

from configs.constants import *


class DCGAN(nn.Module):
    def __init__(self, noise_std=0) -> None:
        super(DCGAN, self).__init__()

        self.noise_std = noise_std

        layers = [
            self._make_layer(N_Z, N_GENERATOR_FEATURES * 4, kernel_size=4, stride=1, padding=0),
            self._make_layer(N_GENERATOR_FEATURES * 4, N_GENERATOR_FEATURES * 2),
            self._make_layer(N_GENERATOR_FEATURES * 2, N_GENERATOR_FEATURES),
            self._make_final_layer(N_GENERATOR_FEATURES, GRID_NUM_CHANNELS),
        ]

        self.model = nn.Sequential(*layers)
        self.alpha = 1000

    def _make_layer(self, iGRID_NUM_CHANNELS, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(iGRID_NUM_CHANNELS, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_final_layer(self, iGRID_NUM_CHANNELS, out_channels):
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


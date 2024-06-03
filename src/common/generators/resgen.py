import torch
import torch.nn as nn

from configs.constants import *


class ResGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.startBlock = nn.Conv2d(GRID_NUM_CHANNELS, 32, kernel_size=3, stride=1, padding=1)

        self.backBone = nn.ModuleList(
            [ResBlock(32) for _ in range(4)]
        )

        self.out_conv = nn.Conv2d(32, GRID_NUM_CHANNELS, kernel_size=1, stride=1, padding=0)

    def generate_log_normal(self, mu, sigma, shape):
        # Generate samples from a normal distribution
        normal_samples = torch.randn(shape) * sigma + mu

        # Apply exponential to convert to log-normal
        log_normal_samples = torch.exp(normal_samples)

        return log_normal_samples

    def forward(self, x):

        # x = self.generate_log_normal(0, 0.1, x.shape)

        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        x = self.out_conv(x)

        x = torch.tanh(x)

        return x

    def name(self):
        str_network = "ResGen"

        return str_network


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


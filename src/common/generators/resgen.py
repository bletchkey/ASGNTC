import torch
import torch.nn as nn

from configs.constants import *

from src.common.utils.toroidal import ToroidalConv2d

class ResGen(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        self.num_hidden = num_hidden

        self.startBlock = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, 32, kernel_size=3, stride=1, padding=0))

        self.backBone = nn.Sequential(
            ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU())

        self.out_conv = nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.startBlock(x)
        x = self.backBone(x)
        x = self.out_conv(x)
        x = torch.tanh(x)

        return x

    def name(self):
        str_network = "ResGen_"
        str_network += f"{self.num_hidden}hidden"

        return str_network


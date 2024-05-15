import torch
import torch.nn as nn

# from src.common.utils.helpers import toroidal_Conv2d
from configs.constants import *


class ResNet(nn.Module):
    def __init__(self, topology, num_resBlocks, num_hidden):
        super().__init__()

        self.topology      = topology
        self.num_resBlocks = num_resBlocks
        self.num_hidden    = num_hidden

        self.start_block = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(GRID_NUM_CHANNELS, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(GRID_NUM_CHANNELS, num_hidden, kernel_size=3, stride=1, padding=1)]
            ),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(num_hidden, GRID_NUM_CHANNELS, kernel_size=1, stride=1, padding=0)

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, self.topology) for _ in range(num_resBlocks)]
        )

    def forward(self, x):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        x = self.out_conv(x)

        return x

    def name(self):
        str_network = "ResNet_"
        str_network += f"{self.num_resBlocks}blocks_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.topology}"

        return str_network

class ResBlock(nn.Module):
    def __init__(self, num_hidden, topology):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            ),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            ),
        )


    def forward(self, x):
        return x + self.block(x)


class ToroidalConv2d(nn.Module):
    def __init__(self, conv2d, padding=1):
        super().__init__()
        self.conv2d  = conv2d
        self.padding = padding

    def forward(self, x):
        x = self.__add_toroidal_padding(x)
        x = self.conv2d(x)
        return x

    def __add_toroidal_padding(self, x):
        if self.padding  <= 0:
            return x

        x = torch.cat([x[:, :, -self.padding :], x, x[:, :, :self.padding ]], dim=2)
        x = torch.cat([x[:, :, :, -self.padding :], x, x[:, :, :, :self.padding ]], dim=3)
        return x


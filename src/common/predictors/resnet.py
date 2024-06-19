import torch
import torch.nn as nn

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


class ResNet(nn.Module):
    def __init__(self, topology, num_resBlocks, num_hidden):
        super().__init__()

        self.topology      = topology
        self.num_resBlocks = num_resBlocks
        self.num_hidden    = num_hidden

        self.startBlock = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )


        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, self.topology) for _ in range(num_resBlocks)]
        )

        self.out_conv = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        x = self.out_conv(x)

        x = torch.sigmoid(x)

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

        conv_layer1 = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)) \
            if topology == TOPOLOGY_TOROIDAL else nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)

        conv_layer2 = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)) \
            if topology == TOPOLOGY_TOROIDAL else nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)


        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer1,
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer2,
        )

    def forward(self, x):
        return x + self.block(x)


import torch
import torch.nn as nn

from configs.constants import *

from src.common.utils.toroidal import ToroidalConv2d


class Gen(nn.Module):
    def __init__(self, topology, num_hidden):
        super().__init__()

        self.num_hidden = num_hidden
        self.topology   = topology

        self.in_conv = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )
        self.in_batch_norm  = nn.BatchNorm2d(num_hidden)

        self.conv_1 = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )
        self.batch_norm_1  = nn.BatchNorm2d(num_hidden)

        self.conv_2 = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )

        )
        self.batch_norm_2 = nn.BatchNorm2d(num_hidden)

        self.conv_3 = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )

        )
        self.batch_norm_3 = nn.BatchNorm2d(num_hidden)


        self.out_conv = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.in_conv(x)
        x = self.in_batch_norm(x)
        x = self.relu(x)

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        x = self.out_conv(x)

        x = torch.tanh(x)

        return x

    def name(self):
        str_network = "Gen_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.topology}"

        return str_network


# class Gen(nn.Module):
#     def __init__(self, num_hidden, num_resBlocks=3):
#         super().__init__()

#         self.num_resBlocks = num_resBlocks
#         self.num_hidden    = num_hidden

#         self.startBlock = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))

#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for _ in range(num_resBlocks)]
#         )

#         self.out_conv = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x = self.startBlock(x)

#         for resBlock in self.backBone:
#             x = resBlock(x)

#         x = self.out_conv(x)

#         x = torch.tanh(x)

#         return x

#     def name(self):
#         str_network = "Gen_"
#         str_network += f"{self.num_resBlocks}blocks_"
#         str_network += f"{self.num_hidden}hidden_Toroidal"

#         return str_network


# class ResBlock(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()

#         conv_layer = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))

#         self.block = nn.Sequential(
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#             conv_layer,
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#             conv_layer,
#         )

#     def forward(self, x):
#         return x + self.block(x)


# # TEST
# class Gen(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()

#         self.num_hidden = num_hidden

#         self.in_conv  = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, NUM_GENERATOR_FEATURES, kernel_size=3, stride=1, padding=0))
#         self.in_batch_norm  = nn.BatchNorm2d(NUM_GENERATOR_FEATURES)

#         self.conv_1  = ToroidalConv2d(nn.Conv2d(NUM_GENERATOR_FEATURES, NUM_GENERATOR_FEATURES, kernel_size=3, stride=1, padding=0))
#         self.batch_norm_1  = nn.BatchNorm2d(NUM_GENERATOR_FEATURES)

#         self.out_conv = nn.Conv2d(NUM_GENERATOR_FEATURES, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

#         self.relu = nn.ReLU()

#     def forward(self, x):

#         x = self.in_conv(x)
#         x = self.in_batch_norm(x)
#         x = self.relu(x)

#         x = self.conv_1(x)
#         x = self.batch_norm_1(x)
#         x = self.relu(x)

#         x = self.out_conv(x)

#         x = torch.tanh(x)

#         return x

#     def name(self):
#         str_network = "Gen_"
#         str_network += f"{self.num_hidden}hidden"

#         return str_network


# class Gen(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()

#         self.num_hidden = num_hidden

#         self.startBlock = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, NUM_GENERATOR_FEATURES, kernel_size=3, stride=1, padding=0))

#         self.backBone = nn.Sequential(
#             ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#             ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#             ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#             ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0)),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU())

#         self.out_conv = nn.Conv2d(NUM_GENERATOR_FEATURES, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):

#         x = self.startBlock(x)
#         x = self.backBone(x)
#         x = self.out_conv(x)
#         x = torch.tanh(x)

#         return x

#     def name(self):
#         str_network = "Gen_"
#         str_network += f"{self.num_hidden}hidden"

#         return str_network


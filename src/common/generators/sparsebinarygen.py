import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.constants import *
from src.common.utils.toroidal import ToroidalConv2d


class SparseBinaryGenerator(nn.Module):
    def __init__(self,
                 topology,
                 num_hidden,
                 temperature=1.0):
        super(SparseBinaryGenerator, self).__init__()

        self.temperature = temperature
        self.num_hidden  = num_hidden
        self.topology    = topology

        self.conv1  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv2  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv3  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        # self.conv4 = nn.Conv2d(num_hidden, 2, kernel_size=1, stride=1, padding=0) logits

        self.conv4 = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        logits = self.conv4(x)

        return logits

        # # Flatten logits for easier processing
        # logits = logits.view(batch_size, 2, -1)  # (batch_size, 2, grid_size*grid_size)
        # probs = F.softmax(logits, dim=1)[:, 1]  # Probabilities for class 1

        # return probs.view(batch_size, 1, GRID_SIZE, GRID_SIZE)

        # # Soft assignment for backpropagation
        # if self.training:
        #     return probs.view(batch_size, 1, GRID_SIZE, GRID_SIZE)

        # # Top-k selection to ensure exactly 32 ones if possible during evaluation
        # topk_values, topk_indices = torch.topk(probs, 32, dim=-1, largest=True)
        # y = torch.zeros_like(probs)
        # y.scatter_(1, topk_indices, 1)

        # y = y.view(batch_size, 1, GRID_SIZE, GRID_SIZE)  # Reshape back to grid shape
        # return y

    def name(self):
        str_network = "SparseBinaryGen_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.topology}"

        return str_network


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from configs.constants import *

class GamblerGumble(nn.Module):
    def __init__(self):
        super(GamblerGumble, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(GRID_NUM_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, GRID_NUM_CHANNELS, kernel_size=3, stride=1, padding=1)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        for conv in self.convs:
            x = conv(x)

        x_flat = self.softmax(x.view(batch_size, -1))

        # Generate a continuous probability distribution using Gumbel-Softmax
        gumbel_output = self.gumbel_softmax(x_flat, tau=1.0, dim=1)

        # Select N_LIVING_CELLS_INITIAL pixels by choosing the top probabilities
        values, indices = torch.topk(gumbel_output, N_LIVING_CELLS_INITIAL, dim=1)

        # Create a new zero tensor and set the selected indices to 1
        y = torch.zeros_like(x_flat)
        y.scatter_(1, indices, 1)

        log_probability = torch.log(values).sum(dim=1)

        # Reshape the output to have the correct dimensions
        y = y.view(batch_size, GRID_NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

        return y, -log_probability


    def gumbel_softmax(self, logits, tau=1.0, dim=-1):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
        gumbel_logits = (logits + gumbels) / tau
        y_soft = F.softmax(gumbel_logits, dim=dim)
        return y_soft


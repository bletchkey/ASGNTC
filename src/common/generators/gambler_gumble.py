import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from configs.constants import *

class GamblerGumble(nn.Module):
    def __init__(self):
        super(GamblerGumble, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, N_CHANNELS, kernel_size=3, stride=1, padding=1)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probability = torch.zeros(x.size(0), device=x.device)

        for conv in self.convs:
            x = conv(x)

        # Flatten and apply softmax
        x = self.softmax(x.view(x.size(0), -1))

        # Apply Gumbel-Softmax
        gumbel_output = self.gumbel_softmax(x, tau=1.0, dim=1)

        # Sample categories based on Gumbel-Softmax output
        _, category = gumbel_output.max(dim=1)

        # Assuming you need to log the probability of selected actions
        log_probability = torch.log(gumbel_output[range(gumbel_output.shape[0]), category])

        # Convert the sample indices into one-hot encoded form and reshape
        y = F.one_hot(category, num_classes=GRID_SIZE * GRID_SIZE).to(x.dtype)
        y = y.view(-1, N_CHANNELS, GRID_SIZE, GRID_SIZE)

        return y, -log_probability

    def gumbel_softmax(self, logits, tau=1.0, dim=-1):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
        gumbel_logits = (logits + gumbels) / tau
        y_soft = F.softmax(gumbel_logits, dim=dim)
        return y_soft


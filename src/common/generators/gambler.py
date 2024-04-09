import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from configs.constants import *


class Gambler(nn.Module):
    def __init__(self) -> None:
        super(Gambler, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(N_CHANNELS, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, N_CHANNELS, kernel_size=3, stride=1, padding=1)
        ])

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:


        log_probability = torch.zeros(x.size(0), device=x.device)

        for n in range(N_LIVING_CELLS_INITIAL):

            for conv in self.convs:
                x = conv(x)

            x = self.softmax(x.view(x.size(0), -1)).view_as(x)
            distribution = torch.distributions.Categorical(x.view(x.size(0), -1))

            category = distribution.sample()
            log_probability += distribution.log_prob(category)

        return x, -log_probability


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from configs.constants import *


class Gambler(nn.Module):
    def __init__(self) -> None:
        super(Gambler, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32,         32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, N_CHANNELS, kernel_size=3, stride=1, padding=1)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        log_probability = torch.zeros(x.size(0), device=x.device)

        for conv in self.convs:
            x = conv(x)

        x = self.softmax(x.view(x.size(0), -1)).view_as(x)

        distribution = torch.distributions.Categorical(x.view(x.size(0), -1))
        category = distribution.sample(torch.Size([N_LIVING_CELLS_INITIAL]))

        log_probability = distribution.log_prob(category).sum(0)

        y = torch.zeros_like(x.view(x.size(0), -1))
        for i in range(N_LIVING_CELLS_INITIAL):
            y.scatter_(1, category[i].unsqueeze(1), 1)

        x = y.view_as(x)

        return x, -log_probability


    # def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:


    #     log_probability = torch.zeros(x.size(0), device=x.device)

    #     for n in range(N_LIVING_CELLS_INITIAL):

    #         for conv in self.convs:
    #             x = conv(x)

    #         x = self.softmax(x.view(x.size(0), -1)).view_as(x)
    #         distribution = torch.distributions.Categorical(x.view(x.size(0), -1))

    #         category = distribution.sample()
    #         log_probability += distribution.log_prob(category)

    #     return x, -log_probability


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from src.common.utils.helpers import toroidal_Conv2d

from configs.constants import *


class Gambler(nn.Module):
    def __init__(self) -> None:
        super(Gambler, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(32,         32, kernel_size=3, stride=1, padding=0),
        ])
        self.last_conv = nn.Conv2d(32, N_CHANNELS, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        log_probability = torch.zeros(x.size(0), device=x.device)

        for conv in self.convs:
            x = toroidal_Conv2d(x, conv, padding=1)
            x = self.relu(x)

        x = self.last_conv(x)
        x = self.relu(x)

        x = self.softmax(x.view(x.size(0), -1)).view_as(x)

        values, indices = torch.topk(x.view(x.size(0), -1), N_LIVING_CELLS_INITIAL)

        y = torch.zeros_like(x.view(x.size(0), -1))
        y.scatter_(1, indices, 1)

        log_probability = torch.log(values).sum(dim=1)

        x = y.view_as(x)

        return x, -log_probability


class Gambler_v2(nn.Module):
    def __init__(self) -> None:
        super(Gambler_v2, self).__init__()

        self.in_conv   = nn.Conv2d(N_CHANNELS,  8, kernel_size=3, stride=1, padding=0)
        self.conv_1    = nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0)
        self.conv_2    = nn.Conv2d(16,         32, kernel_size=3, stride=1, padding=0)
        self.out_conv  = nn.Conv2d(32, N_CHANNELS, kernel_size=1, stride=1, padding=0)

        self.linear_1  = nn.Linear(64, GRID_SIZE * GRID_SIZE)

        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax   = nn.Softmax(dim=1)
        self.relu      = nn.ReLU()

        self.bn1       = nn.BatchNorm2d(8)
        self.bn2       = nn.BatchNorm2d(16)
        self.bn3       = nn.BatchNorm2d(32)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        config          = torch.zeros_like(x)
        probabilities = torch.zeros(batch_size, device=x.device)

        x = toroidal_Conv2d(x, self.in_conv, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = toroidal_Conv2d(x, self.conv_1, padding=1)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = toroidal_Conv2d(x, self.conv_2, padding=1)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.out_conv(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.linear_1(x)

        softmax           = self.softmax(x)
        indices           = torch.multinomial(softmax, N_LIVING_CELLS_INITIAL)
        log_probabilities = -1 * torch.log(softmax.gather(1, indices)).sum(dim=1)

        probabilities     = torch.gather(softmax, 1, indices).sum(dim=1)

        config = torch.scatter(config.view(batch_size, -1), 1, indices, 1).view_as(config)

        return config, probabilities


class Gambler_v3(nn.Module):
    def __init__(self) -> None:
        super(Gambler_v3, self).__init__()

        self.in_conv   = nn.Conv2d(N_CHANNELS,  8, kernel_size=3, stride=1, padding=0)
        self.conv_1    = nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0)
        self.out_conv  = nn.Conv2d(16, N_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        config = torch.zeros_like(x)
        mask   = torch.ones_like(x.view(batch_size, -1))

        for _ in range(N_LIVING_CELLS_INITIAL):

            x = toroidal_Conv2d(x, self.in_conv, padding=1)
            x = self.relu(x)

            x = toroidal_Conv2d(x, self.conv_1, padding=1)
            x = self.relu(x)
            x = self.pool(x)

            x = self.out_conv(x)
            x = self.relu(x)

            x = torch.flatten(x, 1)
            x = self.linear_1(x)

            softmax          = self.softmax(x)
            masked_softmax   = softmax * mask

            index            = torch.multinomial(masked_softmax, 1)
            log_probability += torch.log(masked_softmax.gather(1, index)).squeeze()

            mask = mask.scatter(1, index, 0)

            y = torch.zeros_like(x)
            y = y.scatter_(1, index, 1)

            config += y.view_as(config)
            x       = config

        return config, -log_probability


class Gambler_v4(nn.Module):
    def __init__(self) -> None:
        super(Gambler_v4, self).__init__()

        self.in_conv   = nn.Conv2d(N_CHANNELS,  8, kernel_size=3, stride=1, padding=0)
        self.conv_1    = nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0)
        self.out_conv  = nn.Conv2d(16, N_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        config = torch.zeros_like(x)
        mask   = torch.ones_like(x.view(batch_size, -1))

        for _ in range(N_LIVING_CELLS_INITIAL):

            x = toroidal_Conv2d(x, self.in_conv, padding=1)
            x = self.relu(x)

            x = toroidal_Conv2d(x, self.conv_1, padding=1)
            x = self.relu(x)
            x = self.pool(x)

            x = self.out_conv(x)
            x = self.relu(x)

            x = torch.flatten(x, 1)
            x = self.linear_1(x)

            softmax             = self.softmax(x)
            masked_softmax      = softmax * mask
            highest_prob, index = torch.max(masked_softmax, 1)
            log_probability    += torch.log(highest_prob)

            mask = mask.scatter(1, index.view(-1, 1), 0)

            y = torch.zeros_like(x)
            y = y.scatter_(1, index.view(-1, 1), 1)

            config += y.view_as(config)
            x       = config

        return config, -log_probability


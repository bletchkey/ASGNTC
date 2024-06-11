import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from typing import Tuple

from src.common.utils.toroidal import ToroidalConv2d

from configs.constants import *


import torch
import torch.nn as nn
import random

from configs.constants import *



class DCGAN(nn.Module):
    def __init__(self) -> None:
        super(DCGAN, self).__init__()

        layers = [
            self._make_layer(LATENT_VEC_SIZE, NUM_GENERATOR_FEATURES * 4, kernel_size=4, stride=1, padding=0),
            self._make_layer(NUM_GENERATOR_FEATURES * 4, NUM_GENERATOR_FEATURES * 2),
            self._make_layer(NUM_GENERATOR_FEATURES * 2, NUM_GENERATOR_FEATURES),
            self._make_final_layer(NUM_GENERATOR_FEATURES, NUM_CHANNELS_GRID),
        ]

        self.model = nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_final_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=True),
        )

    def forward(self, input):
        x = self.model(input)

        # # Scale the output to the range [-2, 1]
        # x = 1.5 * nn.Tanh()(x) - 0.5

        x = nn.Tanh()(x)
        return x

    def name(self):
        return "Generator_DCGAN"


class DCGAN_noisy(nn.Module):
    def __init__(self, noise_std=0) -> None:
        super(DCGAN_noisy, self).__init__()

        self.noise_std = noise_std

        layers = [
            self._make_layer(LATENT_VEC_SIZE, NUM_GENERATOR_FEATURES * 4, kernel_size=4, stride=1, padding=0),
            self._make_layer(NUM_GENERATOR_FEATURES * 4, NUM_GENERATOR_FEATURES * 2),
            self._make_layer(NUM_GENERATOR_FEATURES * 2, NUM_GENERATOR_FEATURES),
            self._make_final_layer(NUM_GENERATOR_FEATURES, NUM_CHANNELS_GRID),
        ]

        self.model = nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_final_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=True),
        )

    def threshold(self, x, alpha):
        return (1 + nn.Tanh()(alpha * x)) / 2

    def forward(self, input):
        x = self.model(input)

        # Add variance to the output
        if self.noise_std > 0:
            # mean_shift = random.uniform(-8, 8)
            mean_shift = random.normalvariate(0, 6)
            noise = (torch.randn_like(x) * self.noise_std) + mean_shift
            x = x + noise

        # alpha = 1000
        # x = self.threshold(x, alpha)

        x = nn.Tanh()(x)

        return x

    def name(self):
        return "Generator_DCGAN_noisy"




class Gambler(nn.Module):
    def __init__(self) -> None:
        super(Gambler, self).__init__()

        self.convs = nn.ModuleList([
            ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, 32, kernel_size=3, stride=1, padding=0)),
            ToroidalConv2d(nn.Conv2d(32,         32, kernel_size=3, stride=1, padding=0)),
        ])
        self.last_conv = nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        log_probability = torch.zeros(x.size(0), device=x.device)

        for conv in self.convs:
            x = conv(x)
            x = self.relu(x)

        x = self.last_conv(x)
        x = self.relu(x)

        x = self.softmax(x.view(x.size(0), -1)).view_as(x)

        values, indices = torch.topk(x.view(x.size(0), -1), NUM_LIVING_CELLS_INITIAL)

        y = torch.zeros_like(x.view(x.size(0), -1))
        y.scatter_(1, indices, 1)

        log_probability = torch.log(values).sum(dim=1)

        x = y.view_as(x)

        return x, -log_probability


class Gambler_v2(nn.Module):
    def __init__(self) -> None:
        super(Gambler_v2, self).__init__()

        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.conv_2    = ToroidalConv2d(nn.Conv2d(16,         32, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

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

        x = self.in_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.out_conv(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.linear_1(x)

        softmax           = self.softmax(x)
        indices           = torch.multinomial(softmax, NUM_LIVING_CELLS_INITIAL)
        log_probabilities = -1 * torch.log(softmax.gather(1, indices)).sum(dim=1)

        config = torch.scatter(config.view(batch_size, -1), 1, indices, 1).view_as(config)

        return config, log_probabilities


class Gambler_v3(nn.Module):
    def __init__(self) -> None:
        super(Gambler_v3, self).__init__()

        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(16, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        config = torch.zeros_like(x)
        mask   = torch.ones_like(x.view(batch_size, -1))

        for _ in range(NUM_LIVING_CELLS_INITIAL):

            x = self.in_conv(x)
            x = self.relu(x)

            x = self.conv_1(x)
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

        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(16, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        config = torch.zeros_like(x)
        mask   = torch.ones_like(x.view(batch_size, -1))

        for _ in range(NUM_LIVING_CELLS_INITIAL):

            x = self.in_conv(x)
            x = self.relu(x)

            x = self.conv_1(x)
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


class GamblerBernoullian(nn.Module):
    def __init__(self) -> None:
        super(GamblerBernoullian, self).__init__()

        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(16, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)  # Check if 256 is correct

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        x = self.in_conv(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.out_conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_1(x)

        probabilities = torch.sigmoid(x)  # Convert logits to probabilities
        distribution  = Bernoulli(probabilities)
        samples       = distribution.sample()  # Sample from the Bernoulli distribution

        # Compute log probabilities of the samples, necessary for learning
        log_probs = distribution.log_prob(samples)
        log_probability += log_probs.sum(1)  # Sum across pixels for each batch

        samples = samples.view(batch_size, NUM_CHANNELS_GRID, GRID_SIZE, GRID_SIZE)

        return samples, -log_probability


class GamblerGumble(nn.Module):
    def __init__(self) -> None:
        super(GamblerGumble, self).__init__()

        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(16, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)
        self.linear_1  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size      = x.size(0)
        log_probability = torch.zeros(batch_size, device=x.device)

        config = torch.zeros_like(x)
        mask   = torch.ones_like(x.view(batch_size, -1))

        for _ in range(NUM_LIVING_CELLS_INITIAL):

            x = self.in_conv(x)
            x = self.relu(x)

            x = self.conv_1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.out_conv(x)
            x = self.relu(x)

            x = torch.flatten(x, 1)
            x = self.linear_1(x)

            gumbel_softmax   = self.gumbel_softmax(x, tau=0.5, dim=1)
            masked_softmax   = gumbel_softmax * mask

            index            = torch.multinomial(masked_softmax, 1)
            log_probability += torch.log(masked_softmax.gather(1, index)).squeeze()

            mask = mask.scatter(1, index, 0)

            y = torch.zeros_like(x)
            y = y.scatter_(1, index, 1)

            config += y.view_as(config)
            x       = config

        return config, -log_probability

    def gumbel_softmax(self, logits, tau=1.0, dim=-1):
        gumbels       = -torch.empty_like(logits).exponential_().log()
        gumbel_logits = (logits + gumbels) / tau
        y_soft        = F.softmax(gumbel_logits, dim=dim)
        return y_soft


class GamblerGumble_v2(nn.Module):
    def __init__(self):
        super(GamblerGumble_v2, self).__init__()
        self.in_conv   = ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID,  8, kernel_size=3, stride=1, padding=0))
        self.conv_1    = ToroidalConv2d(nn.Conv2d(8,          16, kernel_size=3, stride=1, padding=0))
        self.conv_2    = ToroidalConv2d(nn.Conv2d(16,         32, kernel_size=3, stride=1, padding=0))
        self.out_conv  = nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=1, stride=1, padding=0)

        self.linear_1  = nn.Linear(64, 256)
        self.linear_2  = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu      = nn.ReLU()

        self.bn1       = nn.BatchNorm2d(8)
        self.bn2       = nn.BatchNorm2d(16)
        self.bn3       = nn.BatchNorm2d(32)

    def forward(self, x):

        batch_size      = x.size(0)

        x = self.in_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.out_conv(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)

        gumbel_softmax = self.gumbel_softmax(x, tau=0.5, dim=1)

        # Choosing the best actions
        values, indices = torch.topk(gumbel_softmax, NUM_LIVING_CELLS_INITIAL, dim=1)

        y = torch.zeros_like(x)
        y = y.scatter_(1, indices, 1)

        log_probability = torch.log(values).sum(dim=1)

        y = y.view(batch_size, NUM_CHANNELS_GRID, GRID_SIZE, GRID_SIZE)

        return y, -log_probability

    def gumbel_softmax(self, logits, tau=1.0, dim=-1):
        gumbels       = -torch.empty_like(logits).exponential_().log()
        gumbel_logits = (logits + gumbels) / tau
        y_soft        = F.softmax(gumbel_logits, dim=dim)
        return y_soft


class GamblerGumble_v3(nn.Module):
    def __init__(self):
        super(GamblerGumble_v3, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(NUM_CHANNELS_GRID, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=3, stride=1, padding=1)
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

        # Select NUM_LIVING_CELLS_INITIAL pixels by choosing the top probabilities
        values, indices = torch.topk(gumbel_output, NUM_LIVING_CELLS_INITIAL, dim=1)

        # Create a new zero tensor and set the selected indices to 1
        y = torch.zeros_like(x_flat)
        y.scatter_(1, indices, 1)

        log_probability = torch.log(values).sum(dim=1)

        # Reshape the output to have the correct dimensions
        y = y.view(batch_size, NUM_CHANNELS_GRID, GRID_SIZE, GRID_SIZE)

        return y, -log_probability


    def gumbel_softmax(self, logits, tau=1.0, dim=-1):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
        gumbel_logits = (logits + gumbels) / tau
        y_soft = F.softmax(gumbel_logits, dim=dim)
        return y_soft


class GamblerGumble(nn.Module):
    def __init__(self):
        super(GamblerGumble, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(NUM_CHANNELS_GRID, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, NUM_CHANNELS_GRID, kernel_size=3, stride=1, padding=1)
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

        # Select NUM_LIVING_CELLS_INITIAL pixels by choosing the top probabilities
        values, indices = torch.topk(gumbel_output, NUM_LIVING_CELLS_INITIAL, dim=1)

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


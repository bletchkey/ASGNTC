import torch
import torch.nn as nn
import random

from .utils import constants as constants

from .generators.DCGAN import DCGAN

def Generator(noise_std=0):
    generator = DCGAN(noise_std=noise_std)

    # Kaiming initialization
    for layer in generator.modules():
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0)

    return generator


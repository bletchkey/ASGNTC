import torch
import torch.nn as nn


from src.gol_adv_sys.generators.DCGAN import DCGAN
from src.gol_adv_sys.utils import constants as constants

def Generator_DCGAN(noise_std=0):
    generator = DCGAN(noise_std=noise_std)

    # Kaiming initialization
    for layer in generator.modules():
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0)

    return generator


import torch
import torch.nn as nn

from torchsummary import summary

from .utils import constants as constants
from .predictors.Baseline import Baseline
from .predictors.UNet import UNet
from .predictors.ResNet import ResNetConstantChannels, block
from .predictors.VisionTransformer import VisionTransformer


def Predictor_Baseline():
    model = Baseline()
    #summary(model, (constants.nc, constants.grid_size, constants.grid_size))
    print(model)
    return model

def Predictor_ResNet():
    return ResNetConstantChannels(block, [2, 2, 2, 2], constants.grid_size)

def Predictor_UNet():
    return UNet()

def Predictor_ViT():
    img_size = constants.grid_size
    patch_size = 8  # Example: using 8x8 patches
    in_channels = constants.nc
    embed_dim = 768  # Size of each embedding
    num_heads = 12  # Number of attention heads
    num_layers = 12  # Number of transformer blocks
    ff_dim = 2048  # Dimension of the feedforward network

    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                          embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)


    return model


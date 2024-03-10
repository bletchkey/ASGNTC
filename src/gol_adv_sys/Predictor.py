import torch
import torch.nn as nn

from torchsummary import summary

from .utils import constants as constants
from .predictors.Baseline import Baseline, Baseline_v2, Baseline_v3
from .predictors.UNet import UNet
from .predictors.ResNet import ResNetConstantChannels
from .predictors.VisionTransformer import VisionTransformer
from .predictors.GloNet import GloNet
from .predictors.GoogLeNet import GoogLeNet


def Predictor_Baseline():
    model = Baseline()
    #summary(model, (constants.nc, constants.grid_size, constants.grid_size))
    print(model)
    return model


def Predictor_Baseline_v2():
    model = Baseline_v2()
    print(model)
    return model


def Predictor_Baseline_v3():
    model = Baseline_v3()
    print(model)
    return model


def Predictor_ResNet():
    module = ResNetConstantChannels([2, 2, 2, 2], constants.grid_size)
    print(module)
    return module


def Predictor_GloNet():
    model = GloNet()
    print(model)
    return model


def Predictor_UNet():
    model = UNet()
    print(model)
    return model


def Predictor_GoogLeNet():
    model = GoogLeNet()
    print(model)
    return model


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

    print(model)
    return model


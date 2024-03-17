import logging

from src.gol_adv_sys.utils import constants as constants
from src.gol_adv_sys.predictors.Baseline import Baseline, Baseline_v2, Baseline_v3, Baseline_v4
from src.gol_adv_sys.predictors.UNet import UNet
from src.gol_adv_sys.predictors.ResNet import ResNetConstantChannels
from src.gol_adv_sys.predictors.VisionTransformer import VisionTransformer
from src.gol_adv_sys.predictors.GloNet import GloNet

def Predictor_Baseline():
    model = Baseline()
    logging.debug(model)
    return model


def Predictor_Baseline_v2():
    model = Baseline_v2()
    logging.debug(model)
    return model


def Predictor_Baseline_v3():
    model = Baseline_v3()
    logging.debug(model)
    return model


def Predictor_Baseline_v4():
    model = Baseline_v4()
    logging.debug(model)
    return model


def Predictor_ResNet():
    module = ResNetConstantChannels([2, 2, 2, 2], constants.grid_size)
    logging.debug(module)
    return module


def Predictor_GloNet():
    model = GloNet()
    logging.debug(model)
    return model


def Predictor_UNet():
    model = UNet()
    logging.debug(model)
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

    logging.debug(model)
    return model


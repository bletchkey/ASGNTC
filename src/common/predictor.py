import logging
import torch
from torch import nn

from configs.constants import *
from src.common.predictors.unet             import UNet
from src.common.predictors.resnet           import ResNet
from src.common.predictors.spatiotemporal   import SpatioTemporal


def Predictor_Baseline(topology):
    num_resBlocks = 10
    num_hidden    = NUM_PREDICTOR_FEATURES
    model = ResNet(topology, num_resBlocks, num_hidden)
    logging.debug(model)

    return model


def Predictor_ResNet(topology, num_resBlocks, num_hidden):
    module = ResNet(topology, num_resBlocks, num_hidden)
    logging.debug(module)

    return module


def Predictor_UNet():
    model = UNet()
    logging.debug(model)

    return model


def Predictor_LifeMotion(topology, num_hidden):

    num_input  = NUM_CHANNELS_GRID
    num_layers = 2
    framesteps = 10

    model = SpatioTemporal(topology,
                           num_input,
                           num_hidden,
                           num_layers,
                           framesteps)
    logging.debug(model)

    return model


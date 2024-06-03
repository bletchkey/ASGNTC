import logging
import torch
from torch import nn

from configs.constants import *
from src.common.predictors.unet             import UNet
from src.common.predictors.resnet           import ResNet
from src.common.predictors.resnet_attention import ResNetAttention


def Predictor_Baseline(topology):
    num_resBlocks = 10
    num_hidden    = 32
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


def Predictor_ResNetAttention(num_hidden):
    model = ResNetAttention(num_hidden)

    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(model.weight)

    logging.debug(model)

    return model


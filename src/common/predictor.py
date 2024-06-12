import logging
import torch
from torch import nn

from configs.constants import *
from src.common.predictors.unet             import UNet
from src.common.predictors.resnet           import ResNet
from src.common.predictors.convlstm         import ConvLSTMPred


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


def Predictor_ConvLSTM(topology, num_hidden):
    model = ConvLSTMPred(topology, num_hidden)
    logging.debug(model)

    return model

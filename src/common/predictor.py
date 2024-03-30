import logging

from configs.constants import *
from src.common.predictors.baseline import Baseline
from src.common.predictors.unet     import UNet
from src.common.predictors.resnet   import ResNetConstantChannels
from src.common.predictors.glonet   import GloNet

def Predictor_Baseline():
    model = Baseline()
    logging.debug(model)

    return model


def Predictor_ResNet():
    module = ResNetConstantChannels([2, 2, 2, 2], GRID_SIZE)
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


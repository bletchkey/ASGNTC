import logging

from config.constants import *
from src.gol_adv_sys.predictors.Baseline import Baseline
from src.gol_adv_sys.predictors.UNet import UNet
from src.gol_adv_sys.predictors.ResNet import ResNetConstantChannels
from src.gol_adv_sys.predictors.GloNet import GloNet

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


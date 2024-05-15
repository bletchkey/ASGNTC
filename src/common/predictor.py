import logging

from configs.constants import *
from src.common.predictors.baseline    import Baseline, Baseline_v2
from src.common.predictors.unet        import UNet
from src.common.predictors.resnet      import ResNet


def Predictor_Baseline():
    model = Baseline()
    logging.debug(model)

    return model


def Predictor_Baseline_v2():
    model = Baseline_v2()
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


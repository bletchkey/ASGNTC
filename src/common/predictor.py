import logging

from configs.constants import *
from src.common.predictors.unet        import UNet
from src.common.predictors.resnet      import ResNet
from src.common.predictors.proposed    import UNetResNetAttention


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

def Predictor_Proposed(num_hidden):
    model = UNetResNetAttention(num_hidden)
    logging.debug(model)

    return model


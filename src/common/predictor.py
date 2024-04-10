import logging

from configs.constants import *
from src.common.predictors.baseline    import Baseline, Baseline_v2
from src.common.predictors.unet        import UNet
from src.common.predictors.resnet      import ResNet
from src.common.predictors.glonet      import GloNet
from src.common.predictors.transformer import GoLTransformer


def Predictor_Baseline():
    model = Baseline()
    logging.debug(model)

    return model


def Predictor_Baseline_v2():
    model = Baseline_v2()
    logging.debug(model)

    return model


def Predictor_ResNet():
    module = ResNet(n_layers=36, channels=GRID_SIZE)
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


def Predictor_Transformer(board_size=GRID_SIZE, num_layers=6, num_heads=8, d_model=512):
    model = GoLTransformer(board_size=board_size, num_layers=num_layers,
                           num_heads=num_heads, d_model=d_model)
    logging.debug(model)

    return model


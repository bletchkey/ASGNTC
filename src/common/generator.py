import logging
import torch
from torch import nn

from src.common.generators.dcgan  import DCGAN
from src.common.generators.gen    import Gen

from configs.constants import *


def Generator_DCGAN():
    model = DCGAN()
    logging.debug(model)

    return model


def Generator_Gen(topology, num_hidden):
    model = Gen(topology, num_hidden)
    logging.debug(model)

    # Kaiming He initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model


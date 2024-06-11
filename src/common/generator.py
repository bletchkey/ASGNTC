import logging
import torch
from torch import nn

from src.common.generators.gen       import Gen
from src.common.generators.binarygen import BinaryGenerator

from configs.constants import *


def Generator_Binary(topology, num_hidden):
    model = BinaryGenerator(topology, num_hidden)
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


def Generator_Baseline(topology):
    model = Gen(topology, NUM_GENERATOR_FEATURES)
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


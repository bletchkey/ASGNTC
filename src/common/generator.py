import logging
import torch
from torch import nn

from src.common.generators.gen             import Gen
from src.common.generators.binarygen       import BinaryGenerator
from src.common.generators.sparsebinarygen import SparseBinaryGenerator

from configs.constants import *


def apply_kaiming_initialization(model):
    """Apply Kaiming He initialization to all conv layers and standard initialization to batch norm layers."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def Generator_Binary(topology, num_hidden):
    model = BinaryGenerator(topology, num_hidden)
    logging.debug(model)

    return model


def Generator_SparseBinary(topology, num_hidden):
    model = SparseBinaryGenerator(topology, num_hidden)
    logging.debug(model)

    return model


def Generator_Baseline(topology):
    model = Gen(topology, NUM_GENERATOR_FEATURES)
    logging.debug(model)

    apply_kaiming_initialization(model)

    return model


def Generator_Gen(topology, num_hidden):
    model = Gen(topology, num_hidden)
    logging.debug(model)

    apply_kaiming_initialization(model)

    return model


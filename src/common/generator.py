import logging

from src.common.generators.dcgan  import DCGAN
from src.common.generators.resgen import ResGen

from configs.constants import *


def Generator_DCGAN():
    model = DCGAN()
    logging.debug(model)

    return model


def Generator_ResGen():
    model = ResGen()
    logging.debug(model)

    return model


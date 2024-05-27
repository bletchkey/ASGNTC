import logging

from src.common.generators.dcgan          import DCGAN

from configs.constants import *


def Generator_DCGAN():
    model = DCGAN()
    logging.debug(model)

    return model


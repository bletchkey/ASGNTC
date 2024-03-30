import logging

from src.common.generators.dcgan import DCGAN
from configs.constants import *


def Generator_DCGAN(noise_std=0):
    model = DCGAN(noise_std=noise_std)
    logging.debug(model)

    return model


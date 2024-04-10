import logging

from src.common.generators.dcgan          import DCGAN
from src.common.generators.gambler        import Gambler, Gambler_v2
from src.common.generators.gambler_gumble import GamblerGumble

from configs.constants import *


def Generator_DCGAN(noise_std=0):
    model = DCGAN(noise_std=noise_std)
    logging.debug(model)

    return model

def Generator_Gambler():
    model = Gambler()
    logging.debug(model)

    return model

def Generator_Gambler_v2():
    model = Gambler_v2()
    logging.debug(model)

    return model

def Generator_Gambler_Gumble():
    model = GamblerGumble()
    logging.debug(model)

    return model


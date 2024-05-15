import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from src.gol_pred_sys.training_pred import TrainingPredictor

from src.common.predictor import Predictor_Baseline, Predictor_Baseline_v2, \
                                 Predictor_ResNet, Predictor_UNet


def train_predictors():
    # 1. Baseline - Toroidal
    num_resBlocks = 20
    num_hidden    = 32
    train_pred    = TrainingPredictor(model=Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden))
    train_pred.run()

    # 2. Baseline - Flat
    num_resBlocks = 20
    num_hidden    = 32
    train_pred    = TrainingPredictor(model=Predictor_ResNet(TOPOLOGY_FLAT, num_resBlocks, num_hidden))

    # 3. ResNet - Toroidal - 10 ResBlocks - 32 hidden
    num_resBlocks = 10
    num_hidden    = 32
    train_pred    = TrainingPredictor(model=Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden))

    # 4. ResNet - Toroidal - 10 ResBlocks - 64 hidden
    num_resBlocks = 10
    num_hidden    = 64
    train_pred    = TrainingPredictor(model=Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden))

    # 5. ResNet - Toroidal - 20 ResBlocks - 64 hidden
    num_resBlocks = 20
    num_hidden    = 64
    train_pred    = TrainingPredictor(model=Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden))

    #6. UNet
    train_pred    = TrainingPredictor(model=Predictor_UNet())
    train_pred.run()

def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    train_predictors()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


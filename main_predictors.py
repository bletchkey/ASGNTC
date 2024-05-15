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

    target_type = CONFIG_METRIC_EASY

    # train_pred   = TrainingPredictor(Predictor_Baseline(),
    #                                  target_type)
    # train_pred.run()


    # 1. Baseline - Toroidal
    num_resBlocks = 10
    num_hidden    = 32
    train_pred    = TrainingPredictor(Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden),
                                      target_type)
    train_pred.run()

    # 2. Baseline - Flat
    num_resBlocks = 10
    num_hidden    = 32
    train_pred    = TrainingPredictor(Predictor_ResNet(TOPOLOGY_FLAT, num_resBlocks, num_hidden),
                                      target_type)
    train_pred.run()

    # # 3. ResNet - Toroidal - 20 ResBlocks - 64 hidden
    # num_resBlocks = 20
    # num_hidden    = 64
    # train_pred    = TrainingPredictor(Predictor_ResNet(TOPOLOGY_TOROIDAL, num_resBlocks, num_hidden),
    #                                   target_type)
    # train_pred.run()

    # # 4. UNet
    # train_pred    = TrainingPredictor(Predictor_UNet(),
    #                                   target_type)
    # train_pred.run()

def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    train_predictors()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


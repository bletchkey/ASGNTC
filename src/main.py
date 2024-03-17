import logging
import logging.config
import json
from pathlib import Path
import sys
import os
import torch


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.paths import PROJECT_NAME, CONFIG_DIR

from src.gol_adv_sys.TrainingPredictor import TrainingPredictor
from src.gol_adv_sys.TrainingAdversarial import TrainingAdversarial
from src.gol_adv_sys.Playground import Playground

from src.gol_adv_sys.Predictor import Predictor_Baseline, Predictor_Baseline_v2, \
                                  Predictor_Baseline_v3, Predictor_Baseline_v4, \
                                  Predictor_ResNet, Predictor_UNet, \
                                  Predictor_GloNet, Predictor_ViT

from src.gol_adv_sys.Generator import Generator_DCGAN

from src.gol_adv_sys.utils import constants as constants


def setup_base_dir():

    dir = Path(__file__).resolve().parent.parent

    if dir.name != PROJECT_NAME:
        print(f"Error: The base directory is not set correctly. Expected: {PROJECT_NAME}, got: {dir.name}")
        sys.exit(1)

    os.chdir(dir)
    sys.path.append(str(dir))


def setup_logging(path, default_level=logging.INFO):
    try:
        with open(path, 'rt') as file:
            config = json.load(file)
        logging.config.dictConfig(config)
        logging.info(f"Logging configuration loaded from {path}")
    except Exception as e:
        logging.error(f"Error in logging configuration (using default settings): {e}")
        logging.basicConfig(level=default_level)


def playground():
    # init_config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    # Set the cells for a glider
    # init_config[0, 0, 15, 16] = 1
    # init_config[0, 0, 16, 17] = 1
    # init_config[0, 0, 17, 15] = 1
    # init_config[0, 0, 17, 16] = 1
    # init_config[0, 0, 17, 17] = 1

    init_config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    n_init_cells = 512
    indices = torch.randperm(32*32)[:n_init_cells]
    rows, cols = indices // 32, indices % 32
    init_config[0, 0, rows, cols] = 1

    pg = Playground(topology=constants.TOPOLOGY_TYPE["toroidal"])

    results = pg.simulate(init_config, steps=10)

    print(f"Period: {results['period'].item()}")
    print(f"Antiperiod: {results['antiperiod'].item()}")
    print(f"Initial living cells: {results['n_cells_init'].item()}")
    print(f"Final living cells: {results['n_cells_final'].item()}")


def train_adversarial():
    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(),
                                    model_g=Generator_DCGAN())
    train_adv.run()


def train_predictor():
    train_pred = TrainingPredictor(model=Predictor_Baseline_v4())
    train_pred.run()


def main():

    setup_base_dir()
    setup_logging(path=CONFIG_DIR / "logging.json")

    train_predictor()
    # train_adversarial()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


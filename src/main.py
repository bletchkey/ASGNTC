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

from src.gol_adv_sys.Predictor import Predictor_Baseline, Predictor_ResNet, \
                                      Predictor_UNet, Predictor_GloNet \

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
        logging.debug(f"Logging configuration loaded from {path}")
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

    pg = Playground()

    #results = pg.simulate(init_config, steps=10)

    # print(f"Period: {results['period'].item()}")
    # print(f"Transient phase: {results['transient_phase'].item()}")
    # print(f"Initial living cells: {results['n_cells_init'].item()}")
    # print(f"Final living cells: {results
    # ['n_cells_final'].item()}")

    data = pg.get_record_from_id(100000)

    pg.plot_record(data)

    # pg.load_predictor("predictor_medium_metric.tar")
    # pred = pg.predict(data["initial_config"])

    # print(pred.max())
    # print(pred.min())


def train_adversarial():
    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(),
                                    model_g=Generator_DCGAN())
    train_adv.run()


def train_predictor():
    train_pred = TrainingPredictor(model=Predictor_Baseline())
    train_pred.run()


def main():

    setup_base_dir()
    setup_logging(path=CONFIG_DIR / "logging.json")

    playground()
    # train_predictor()
    # train_adversarial()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


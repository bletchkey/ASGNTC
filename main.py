import sys
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from src.gol_pred_sys.training_pred import TrainingPredictor
from src.gol_adv_sys.training_adv   import TrainingAdversarial
from src.common.playground          import Playground

from src.common.predictor import Predictor_Baseline, Predictor_ResNet, \
                                      Predictor_UNet, Predictor_GloNet \

from src.common.generator import Generator_DCGAN


def playground():

    init_config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    n_init_cells = 512
    indices = torch.randperm(32*32)[:n_init_cells]
    rows, cols = indices // 32, indices % 32
    init_config[0, 0, rows, cols] = 1

    pg = Playground()

    results = pg.simulate(init_config, steps=10)

    print(f"Stable min: {results[CONFIG_METRIC_STABLE].min()}")
    print(f"Stable max: {results[CONFIG_METRIC_STABLE].max()}")

    data = pg.get_record_from_id(200000)
    pg.plot_record(data)


def train_adversarial():
    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(),
                                    model_g=Generator_DCGAN())
    train_adv.run()


def train_predictor():
    train_pred = TrainingPredictor(model=Predictor_Baseline())
    train_pred.run()


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    # train_predictor()
    train_adversarial()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


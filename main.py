import sys
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from src.gol_pred_sys.training_pred import TrainingPredictor
from src.gol_adv_sys.training_adv   import TrainingAdversarial
from src.common.playground          import Playground

from src.common.predictor import Predictor_Baseline, Predictor_ResNet, \
                                 Predictor_UNet, Predictor_GloNet, \
                                 Predictor_Transformer

from src.common.generator import Generator_DCGAN, Generator_Gambler, Generator_Gambler_Gumble


def playground():

    # init_config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    # n_init_cells = 512
    # indices = torch.randperm(32*32)[:n_init_cells]
    # rows, cols = indices // 32, indices % 32
    # init_config[0, 0, rows, cols] = 1

    # pg = Playground()

    # results = pg.simulate(init_config, steps=10)

    # print(f"Stable min: {results[CONFIG_METRIC_STABLE].min()}")
    # print(f"Stable max: {results[CONFIG_METRIC_STABLE].max()}")

    # data = pg.get_record_from_id(200000)
    # pg.plot_record_db(data)

    pg = Playground()

    # inital_config = pg.ulam_spiral(GRID_SIZE)
    # inital_config = inital_config.unsqueeze(0).unsqueeze(0)
    # results       = pg.simulate(inital_config, steps=1)
    # pg.plot_record_sim(results)

    config, prob = pg.generate_gambler(4)

    for c in config:
        print(c)
        print(torch.argmax(c.view(-1)))
        print(torch.sum(c.view(-1)))

    print(f"Prob: {prob}")


    config, prob = pg.generate_gambler(BATCH_SIZE)
    results = pg.simulate(config, steps=20)
    pg.plot_record_sim(results)


def train_adversarial():
    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(),
                                    model_g=Generator_Gambler_Gumble())
    train_adv.run()


def train_predictor():
    train_pred = TrainingPredictor(model=Predictor_Baseline())
    train_pred.run()


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    # train_predictor()
    train_adversarial()
    # playground()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


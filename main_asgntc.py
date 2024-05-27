import sys

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from src.gol_adv_sys.training_adv   import TrainingAdversarial

from src.common.predictor import Predictor_Baseline, Predictor_ResNet, Predictor_UNet

from src.common.generator import Generator_DCGAN


def train_adversarial():
    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(TOPOLOGY_TOROIDAL),
                                    model_g=Generator_DCGAN())
    train_adv.run()


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_asgntc.json")

    train_adversarial()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


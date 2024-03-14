import sys
import torch
import random


from gol_adv_sys.TrainingPredictor import TrainingPredictor
from gol_adv_sys.TrainingAdversarial import TrainingAdversarial
from gol_adv_sys.Playground import Playground

from gol_adv_sys.Predictor import Predictor_Baseline, Predictor_Baseline_v2, \
                                  Predictor_Baseline_v3, Predictor_Baseline_v4, \
                                  Predictor_ResNet, Predictor_UNet, \
                                  Predictor_GloNet, Predictor_ViT

from gol_adv_sys.Generator import Generator_DCGAN

from gol_adv_sys.utils import constants as constants

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


def main():

    # playground()

    train_pred = TrainingPredictor(model=Predictor_Baseline())
    train_pred.run()

    # train_adv = TrainingAdversarial(model_p=Predictor_Baseline(), model_g=Generator_DCGAN())
    # train_adv.run()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


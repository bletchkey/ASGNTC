import sys

from gol_adv_sys.TrainingPredictor import TrainingPredictor
from gol_adv_sys.TrainingAdversarial import TrainingAdversarial

from gol_adv_sys.Predictor import Predictor_Baseline, Predictor_Baseline_v2, Predictor_Baseline_v3
from gol_adv_sys.Predictor import Predictor_ResNet, Predictor_GoogLeNet, Predictor_UNet
from gol_adv_sys.Predictor import Predictor_GloNet, Predictor_ViT

from gol_adv_sys.Generator import Generator_DCGAN


def main():

    train_pred = TrainingPredictor(model_p=Predictor_Baseline_v2())
    train_pred.run()

    # train_adv = TrainingAdversarial(model_p=Predictor_Baseline(), model_g=Generator_DCGAN())
    # train_adv.run()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR, TRAININGS_DIR

from src.gol_pred_sys.training_pred import TrainingPredictor

from src.common.utils.helpers import retrieve_log_data
from src.common.predictor import Predictor_Baseline, Predictor_ResNet, Predictor_UNet

def plot_data_base_toro_vs_zero(tor_loss_train, tor_loss_val,
                                tor_acc_train, tor_acc_val,
                                zero_loss_train, zero_loss_val,
                                zero_acc_train, zero_acc_val,
                                target_type):


    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f"Toroidal padding vs Zero padding - Baseline Model on target: {target_type}", fontsize=18)

    ax[0].set_title("Loss")
    ax[0].set_yscale('log')

    ax[0].plot(tor_loss_train, label='toroidal training loss', color='blue')
    ax[0].plot(tor_loss_val, label='toroidal validation loss', color='blue', linestyle="--", linewidth=0.8)
    ax[0].plot(zero_loss_train, label='zero training loss', color='green')
    ax[0].plot(zero_loss_val, label='zero validation loss', color='green', linestyle="--", linewidth=0.8)

    ax[0].set_xlabel("Epoch")
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper right')


    ax[1].set_title("Accuracy")
    ax[1].set_yscale('log')
    ax[1].plot(tor_acc_train, label='toroidal training accuracy', color='blue')
    ax[1].plot(tor_acc_val, label='toroidal validation accuracy', color='blue', linestyle="--", linewidth=0.8)
    ax[1].plot(zero_acc_train, label='zero training accuracy', color='green')
    ax[1].plot(zero_acc_val, label='zero validation accuracy', color='green', linestyle="--", linewidth=0.8)

    ax[1].set_xlabel("epoch")
    ax[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("baseline_comparison_toro_vs_zero.png")

def train_baseline(target_type, topology):
    train_pred = TrainingPredictor(Predictor_Baseline(topology), target_type)
    res = train_pred.run()

    return res

def train_unet(target_type):
    train_pred = TrainingPredictor(Predictor_UNet(), target_type)
    res = train_pred.run()

    return res


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    target_type = CONFIG_METRIC_MEDIUM
    # train_baseline(target_type, TOPOLOGY_FLAT)

    data_toro = retrieve_log_data(log_path= TRAININGS_DIR / "ResNet_Baseline_Medium_Toroidal/logs/training_progress.txt")
    data_zero = retrieve_log_data(log_path= TRAININGS_DIR / "2024-05-17_22-04-15/logs/training_progress.txt")

    plot_data_base_toro_vs_zero(data_toro["train_losses"],
                                data_toro["val_losses"],
                                data_toro["train_accuracies"],
                                data_toro["val_accuracies"],
                                data_zero["train_losses"],
                                data_zero["val_losses"],
                                data_zero["train_accuracies"],
                                data_zero["val_accuracies"],
                                target_type)

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


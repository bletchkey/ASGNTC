import sys
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR, TRAININGS_DIR, \
                          TRAINED_MODELS_DIR, TRAININGS_PREDICTOR_DIR,\
                          DATASET_DIR, OUTPUTS_DIR

from src.gol_pred_sys.training_pred import TrainingPredictor
from src.gol_pred_sys.utils.eval    import get_accuracies

from src.common.utils.plotting      import save_pdf

from src.common.utils.helpers       import retrieve_log_data
from src.common.predictor           import Predictor_Baseline, Predictor_ResNet,\
                                           Predictor_UNet, Predictor_Proposed


def plot_baseline_on_all_targets():
    def plot_metrics(ax, train_data, val_data, title, xlabel, ylabel, yscale='linear', ylim=None, legend_loc='upper right'):
        ax.plot(train_data, label=f'{title.lower()} training', color='blue')
        ax.plot(val_data, label=f'{title.lower()} validation', color='blue', linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc=legend_loc)
        ax.grid(True)

    # Retrieve log data for easy and medium targets
    log_paths = {
        "easy"  : TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Easy/logs/training_progress.txt",
        "medium": TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Medium/logs/training_progress.txt",
        "hard"  : TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Hard/logs/training_progress.txt",
        "stable": TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Stable/logs/training_progress.txt"
    }

    data_easy   = retrieve_log_data(log_paths["easy"])
    data_medium = retrieve_log_data(log_paths["medium"])
    data_hard   = retrieve_log_data(log_paths["hard"])

    # Extract training and validation losses and accuracies
    metrics = {
        "easy": {
            "loss_train": data_easy["train_losses"],
            "loss_val": data_easy["val_losses"],
            "acc_train": data_easy["train_accuracies"],
            "acc_val": data_easy["val_accuracies"]
        },
        "medium": {
            "loss_train": data_medium["train_losses"],
            "loss_val": data_medium["val_losses"],
            "acc_train": data_medium["train_accuracies"],
            "acc_val": data_medium["val_accuracies"]
        },
        "hard": {
            "loss_train": data_hard["train_losses"],
            "loss_val": data_hard["val_losses"],
            "acc_train": data_hard["train_accuracies"],
            "acc_val": data_hard["val_accuracies"]
        }
    }

    # Plotting
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    plt.suptitle("Targets - Baseline Model", fontsize=18)

    # Easy Loss
    plot_metrics(ax[0, 0], metrics["easy"]["loss_train"], metrics["easy"]["loss_val"],
                 "Target Easy - Loss", "Epoch", "Loss", yscale='log')

    # Easy Accuracy
    plot_metrics(ax[0, 1], metrics["easy"]["acc_train"], metrics["easy"]["acc_val"],
                 "Target Easy - Accuracy", "Epoch", "Accuracy", ylim=[0, 100], legend_loc='lower right')

    # Medium Loss
    plot_metrics(ax[1, 0], metrics["medium"]["loss_train"], metrics["medium"]["loss_val"],
                 "Target Medium - Loss", "Epoch", "Loss", yscale='log')

    # Medium Accuracy
    plot_metrics(ax[1, 1], metrics["medium"]["acc_train"], metrics["medium"]["acc_val"],
                 "Target Medium - Accuracy", "Epoch", "Accuracy", ylim=[0, 100], legend_loc='lower right')

    # Hard Loss
    plot_metrics(ax[2, 0], metrics["hard"]["loss_train"], metrics["hard"]["loss_val"],
                 "Target Hard - Loss", "Epoch", "Loss", yscale='log')

    # Hard Accuracy
    plot_metrics(ax[2, 1], metrics["hard"]["acc_train"], metrics["hard"]["acc_val"],
                    "Target Hard - Accuracy", "Epoch", "Accuracy", ylim=[0, 100], legend_loc='lower right')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(OUTPUTS_DIR / "new_targets_baseline_model.png")


def plot_data_base_toro_vs_zero():

    data_toro = retrieve_log_data(log_path= TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Medium/logs/training_progress.txt")
    data_zero = retrieve_log_data(log_path= TRAINED_MODELS_DIR / "predictors/Baseline_Zero_Medium/logs/training_progress.txt")

    if data_toro["target_type"] != data_zero["target_type"]:
        print("The target types are different")
        return

    tor_loss_train  = data_toro["train_losses"]
    tor_loss_val    = data_toro["val_losses"]
    tor_acc_train   = data_toro["train_accuracies"]
    tor_acc_val     = data_toro["val_accuracies"]
    zero_loss_train = data_zero["train_losses"]
    zero_loss_val   = data_zero["val_losses"]
    zero_acc_train  = data_zero["train_accuracies"]
    zero_acc_val    = data_zero["val_accuracies"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f"Toroidal padding vs Zero padding - Baseline Model on target: {data_toro['target_type']}", fontsize=18)

    ax[0].plot(tor_loss_train, label='toroidal padding training loss', color='blue')
    ax[0].plot(tor_loss_val, label='toroidal padding validation loss', color='blue', linestyle="--", linewidth=0.8)
    ax[0].plot(zero_loss_train, label='zero padding training loss', color='green')
    ax[0].plot(zero_loss_val, label='zero padding validation loss', color='green', linestyle="--", linewidth=0.8)

    ax[0].set_title("Loss")
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Epoch")
    ax[0].legend(loc='upper right')


    ax[1].plot(tor_acc_train, label='toroidal padding training accuracy', color='blue')
    ax[1].plot(tor_acc_val, label='toroidal padding validation accuracy', color='blue', linestyle="--", linewidth=0.8)
    ax[1].plot(zero_acc_train, label='zero padding training accuracy', color='green')
    ax[1].plot(zero_acc_val, label='zero padding validation accuracy', color='green', linestyle="--", linewidth=0.8)

    ax[1].set_title("Accuracy")
    ax[1].set_yscale('log')
    ax[1].set_xlabel("epoch")
    ax[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "new_baseline_comparison_toro_vs_zero.png")


def plot_accuracies(model_folder_path):

    accuracies = get_accuracies(model_folder_path)

    accuracies = [100 * acc for acc in accuracies]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.suptitle(f"Accuracies on test set", fontsize=18)

    ax.plot(accuracies, label='accuracies', color='blue')
    ax.set_title("Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_yscale('linear')
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right')
    ax.grid(True)

    plt.tight_layout()

    pdf_path = OUTPUTS_DIR / f"accuracies.pdf"
    save_pdf(pdf_path, [fig])



def train_baseline(target_type, topology):
    train_pred = TrainingPredictor(Predictor_Baseline(topology), target_type)
    res = train_pred.run()

    return res

def train_unet(target_type):
    train_pred = TrainingPredictor(Predictor_UNet(), target_type)
    res = train_pred.run()

    return res

def train_proposed(target_type):
    train_pred = TrainingPredictor(Predictor_Proposed(32), target_type)
    res = train_pred.run()

    return res


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_predictors.json")

    # plot_data_base_toro_vs_zero()
    # plot_baseline_on_all_targets()
    # plot_accuracies(TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Medium")

    train_baseline(CONFIG_TARGET_EASY, TOPOLOGY_TOROIDAL)
    # train_baseline(CONFIG_TARGET_MEDIUM, TOPOLOGY_TOROIDAL)
    # train_baseline(CONFIG_TARGET_HARD, TOPOLOGY_TOROIDAL) -
    # train_baseline(CONFIG_TARGET_STABLE, TOPOLOGY_TOROIDAL)

    # train_baseline(CONFIG_TARGET_MEDIUM, TOPOLOGY_FLAT)

    # train_proposed(CONFIG_TARGET_EASY)

    return 0



if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


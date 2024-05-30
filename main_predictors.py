import sys
import logging
import os
import gc
import time
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
from src.gol_pred_sys.utils.eval    import get_prediction_score, get_prediction_score_n_cells_initial

from src.common.utils.helpers       import export_figures_to_pdf, retrieve_log_data,\
                                           get_model_data_from_checkpoint, get_latest_checkpoint_path

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
    data_stable = retrieve_log_data(log_paths["stable"])

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
        },
        "stable": {
            "loss_train": data_stable["train_losses"],
            "loss_val": data_stable["val_losses"],
            "acc_train": data_stable["train_accuracies"],
            "acc_val": data_stable["val_accuracies"]
        }
    }

    # Plotting
    fig, ax = plt.subplots(4, 2, figsize=(12, 10))
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

    # Stable Loss
    plot_metrics(ax[3, 0], metrics["stable"]["loss_train"], metrics["stable"]["loss_val"],
                 "Target Stable - Loss", "Epoch", "Loss", yscale='log')

    # Stable Accuracy
    plot_metrics(ax[3, 1], metrics["stable"]["acc_train"], metrics["stable"]["acc_val"],
                 "Target Stable - Accuracy", "Epoch", "Accuracy", ylim=[0, 100], legend_loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "targets_baseline_model.pdf"
    export_figures_to_pdf(pdf_path, fig)


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

    pdf_path = OUTPUTS_DIR / "new_baseline_comparison_toro_vs_zero.pdf"
    export_figures_to_pdf(pdf_path, fig)


def plot_baseline_pred_score_each_target():

    model_folder_path = TRAINED_MODELS_DIR / "predictors"

    model_easy   = model_folder_path / "Baseline_Toroidal_Easy"
    model_medium = model_folder_path / "Baseline_Toroidal_Medium"
    model_hard   = model_folder_path / "Baseline_Toroidal_Hard"
    model_stable = model_folder_path / "Baseline_Toroidal_Stable"

    avg_easy_scores_each_checkpoint   = get_prediction_score(model_easy)
    avg_medium_scores_each_checkpoint = get_prediction_score(model_medium)
    avg_hard_scores_each_checkpoint   = get_prediction_score(model_hard)
    avg_stable_scores_each_checkpoint = get_prediction_score(model_stable)

    checkpoint_index = 100

    avg_easy_p_score   = get_prediction_score(model_easy, checkpoint_index)
    avg_medium_p_score = get_prediction_score(model_medium, checkpoint_index)
    avg_hard_p_score   = get_prediction_score(model_hard, checkpoint_index)
    avg_stable_p_score = get_prediction_score(model_stable, checkpoint_index)

    model_easy_checkpoint   = model_easy / "checkpoints" / f"checkpoint_{checkpoint_index}.pt"
    model_medium_checkpoint = model_medium / "checkpoints" / f"checkpoint_{checkpoint_index}.pt"
    model_hard_checkpoint   = model_hard / "checkpoints" / f"checkpoint_{checkpoint_index}.pt"
    model_stable_checkpoint = model_stable / "checkpoints" / f"checkpoint_{checkpoint_index}.pt"

    avg_easy_p_scores_each_n_cells   = get_prediction_score_n_cells_initial(model_easy_checkpoint)
    avg_medium_p_scores_each_n_cells = get_prediction_score_n_cells_initial(model_medium_checkpoint)
    avg_hard_p_scores_each_n_cells   = get_prediction_score_n_cells_initial(model_hard_checkpoint)
    avg_stable_p_scores_each_n_cells = get_prediction_score_n_cells_initial(model_stable_checkpoint)


    # Plotting
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle("Baseline Model - Prediction Score Analysis on Test Dataset", fontsize=18, fontweight='bold')


    # EASY
    axs[0, 0].plot(avg_easy_scores_each_checkpoint.keys(), avg_easy_scores_each_checkpoint.values(),
                     label='Average prediction score for each epoch', color='blue')
    axs[0, 0].set_title("Easy - Performance during training")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Score (%)")
    axs[0, 0].legend()

    axs[0, 1].plot(avg_easy_p_scores_each_n_cells.keys(), avg_easy_p_scores_each_n_cells.values(),
                   label='Average prediction score for each number of initial cells', color='blue')
    axs[0, 1].plot(np.arange(0, 1025), [avg_easy_p_score]*1025,
                   label='Average prediction score', color='red', linestyle="--", linewidth=0.8)
    axs[0, 1].set_title(f"Easy - Performance on each number of initial cells - Epoch {checkpoint_index}")
    axs[0, 1].set_xlabel("Number of initial cells")
    axs[0, 1].set_ylabel("Score (%)")
    axs[0, 1].legend()


    # MEDIUM
    axs[1, 0].plot(avg_medium_scores_each_checkpoint.keys(), avg_medium_scores_each_checkpoint.values(),
                     label='Average prediction score for each epoch', color='blue')
    axs[1, 0].set_title("Medium - Performance during training")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Score (%)")
    axs[1, 0].legend()

    axs[1, 1].plot(avg_medium_p_scores_each_n_cells.keys(), avg_medium_p_scores_each_n_cells.values(),
                   label='Average prediction score for each number of initial cells', color='blue')
    axs[1, 1].plot(np.arange(0, 1025), [avg_medium_p_score]*1025,
                   label='Average prediction score', color='red', linestyle="--", linewidth=0.8)
    axs[1, 1].set_title(f"Medium - Performance on each number of initial cells - Epoch {checkpoint_index}")
    axs[1, 1].set_xlabel("Number of initial cells")
    axs[1, 1].set_ylabel("Score (%)")
    axs[1, 1].legend()


    # HARD
    axs[2, 0].plot(avg_hard_scores_each_checkpoint.keys(), avg_hard_scores_each_checkpoint.values(),
                        label='Average prediction score for each epoch', color='blue')
    axs[2, 0].set_title("Hard - Performance during training")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Score (%)")
    axs[2, 0].legend()

    axs[2, 1].plot(avg_hard_p_scores_each_n_cells.keys(), avg_hard_p_scores_each_n_cells.values(),
                   label='Average prediction score for each number of initial cells', color='blue')
    axs[2, 1].plot(np.arange(0, 1025), [avg_hard_p_score]*1025,
                   label='Average prediction score', color='red', linestyle="--", linewidth=0.8)
    axs[2, 1].set_title(f"Hard - Performance on each number of initial cells - Epoch {checkpoint_index}")
    axs[2, 1].set_xlabel("Number of initial cells")
    axs[2, 1].set_ylabel("Score (%)")
    axs[2, 1].legend()


    # STABLE
    axs[3, 0].plot(avg_stable_scores_each_checkpoint.keys(), avg_stable_scores_each_checkpoint.values(),
                        label='Average prediction score for each epoch', color='blue')
    axs[3, 0].set_title("Stable - Performance during training")
    axs[3, 0].set_xlabel("Epoch")
    axs[3, 0].set_ylabel("Score (%)")
    axs[3, 0].legend()

    axs[3, 1].plot(avg_stable_p_scores_each_n_cells.keys(), avg_stable_p_scores_each_n_cells.values(),
                   label='Average prediction score for each number of initial cells', color='blue')
    axs[3, 1].plot(np.arange(0, 1025), [avg_stable_p_score]*1025,
                   label='Average prediction score', color='red', linestyle="--", linewidth=0.8)
    axs[3, 1].set_title(f"Stable - Performance on each number of initial cells - Epoch {checkpoint_index}")
    axs[3, 1].set_xlabel("Number of initial cells")
    axs[3, 1].set_ylabel("Score (%)")
    axs[3, 1].legend()

    plt.tight_layout()
    pdf_path = OUTPUTS_DIR / "baseline_avg_pred_score_each_target.pdf"
    export_figures_to_pdf(pdf_path, fig)


def train(predictor, target_type, max_retries=5):
    attempts = 0
    while attempts < max_retries:
        try:
            train_pred = TrainingPredictor(predictor, target_type)
            result = train_pred.run()
            del train_pred
            torch.cuda.empty_cache()
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Caught an out of memory error on attempt {attempts + 1}. Trying to clear CUDA cache and retry...")
                torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
                time.sleep(5)  # Wait a bit for memory to clear
                attempts += 1
            else:
                raise e  # If error is not memory related, raise it
    raise RuntimeError("Failed to complete training after several retries due to memory issues.")


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_predictors.json")

    # plot_data_base_toro_vs_zero()
    # plot_baseline_on_all_targets()

    # plot_baseline_pred_score_each_target()

    train(Predictor_Baseline(TOPOLOGY_TOROIDAL), CONFIG_TARGET_STABLE)

    train(Predictor_Baseline(TOPOLOGY_FLAT), CONFIG_TARGET_MEDIUM)

    train(Predictor_Baseline(TOPOLOGY_TOROIDAL), CONFIG_TARGET_MEDIUM)

    # train(Predictor_ResNet(TOPOLOGY_TOROIDAL, 10, 64), CONFIG_TARGET_MEDIUM)

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


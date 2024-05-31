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


def __plot_trainings(ax, train_data, val_data, title, xlabel, ylabel, yscale='linear', ylim=None, legend_loc='upper right'):
        # Use distinct, easily distinguishable colors
        train_color = 'darkblue'
        val_color   = 'crimson'

        # Plot data with markers and different line styles
        ax.plot(train_data, label='training', color=train_color, marker='o', markersize=1.5)
        ax.plot(val_data, label='validation', color=val_color, linestyle="--", linewidth=0.5, marker='x', markersize=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)

        # Automatically adjust ylim based on data if not provided
        if ylim is None:
            all_data = train_data + val_data
            buffer = (max(all_data) - min(all_data)) * 0.1
            ylim = [min(all_data) - buffer, max(all_data) + buffer]
        ax.set_ylim(ylim)
        print(ylim)

        ax.legend(loc=legend_loc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Annotate important points
        min_val_epoch  = val_data.index(min(val_data))
        max_val_epoch  = val_data.index(max(val_data))
        last_val_epoch = len(val_data) - 1

        if yscale == 'log':
            ax.annotate(f'Min: {min(val_data):.3f}', (min_val_epoch, min(val_data)), textcoords="offset points", xytext=(10,20),    ha='center',    color=val_color, fontsize=8)
            ax.annotate(f'Max: {max(val_data):.3f}', (max_val_epoch, max(val_data)), textcoords="offset points", xytext=(25,0),    ha='center',   color=val_color, fontsize=8)

            if (val_data[-1] >= 90):
                pos_y = -15
            else:
                pos_y = 8

            ax.annotate(f'Last: {val_data[-1]:.3f}', (last_val_epoch, val_data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=val_color, fontsize=8)

        else:
            ax.annotate(f'Min: {min(val_data):.1f}%', (min_val_epoch, min(val_data)), textcoords="offset points", xytext=(15,20),    ha='center',    color=val_color, fontsize=8)
            ax.annotate(f'Max: {max(val_data):.1f}%', (max_val_epoch, max(val_data)), textcoords="offset points", xytext=(5,-20),    ha='center',   color=val_color, fontsize=8)

            if (val_data[-1] >= 90):
                pos_y = -15
            else:
                pos_y = 8

            ax.annotate(f'Last: {val_data[-1]:.1f}%', (last_val_epoch, val_data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=val_color, fontsize=8)


def __plot_toro_zero(ax, data, title, xlabel, ylabel, yscale='linear', ylim=None, legend_loc='upper right'):
        # Use distinct, easily distinguishable colors
        toro_color = '#385BA8'
        zero_color = '#366926'

        toro_data = data["toro"]
        zero_data = data["zero"]

        # Plot data with markers and different line styles
        ax.plot(toro_data["train"], label='toroidal padding', color=toro_color, marker='o', markersize=1.5)
        ax.plot(zero_data["train"], label='zero padding', color=zero_color, marker='o', markersize=1.5)
        ax.plot(toro_data["val"], color=toro_color, linestyle='--', linewidth=0.5)
        ax.plot(zero_data["val"], color=zero_color, linestyle='--', linewidth=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)

        # Automatically adjust ylim based on data if not provided
        if ylim is None:
            all_data = toro_data["train"] + toro_data["val"] + zero_data["train"] + zero_data["val"]
            buffer = (max(all_data) - min(all_data)) * 0.1
            ylim = [min(all_data) - buffer, max(all_data) + buffer]
        ax.set_ylim(ylim)
        print(ylim)

        ax.legend(loc=legend_loc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Annotate important points
        for data in [toro_data["val"], zero_data["val"]]:

            color = toro_color if data == toro_data["val"] else zero_color
            min_val_epoch  = data.index(min(data))
            max_val_epoch  = data.index(max(data))
            last_val_epoch = len(data) - 1

            if yscale == 'log':
                ax.annotate(f'Min: {min(data):.3f}', (min_val_epoch, min(data)), textcoords="offset points", xytext=(10,20),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.3f}', (max_val_epoch, max(data)), textcoords="offset points", xytext=(25,0),    ha='center',   color=color, fontsize=8)

                if (data[-1] >= 90):
                    pos_y = -15
                else:
                    pos_y = 8

                ax.annotate(f'Last: {data[-1]:.3f}', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=color, fontsize=8)

            else:
                ax.annotate(f'Min: {min(data):.1f}%', (min_val_epoch, min(data)), textcoords="offset points", xytext=(15,20),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.1f}%', (max_val_epoch, max(data)), textcoords="offset points", xytext=(5,-30),    ha='center',   color=color, fontsize=8)

                if (data[-1] >= 90):
                    pos_y = -15
                else:
                    pos_y = 8

                ax.annotate(f'Last: {data[-1]:.1f}%', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=color, fontsize=8)


def plot_baseline_on_all_targets():

    # Retrieve log data for easy and medium targetss
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
    fig, ax = plt.subplots(4, 2, figsize=(10, 12))
    plt.suptitle("Baseline Model - Training", fontsize=18, fontweight='bold')

    # Easy Loss
    __plot_trainings(ax[0, 0], metrics["easy"]["loss_train"], metrics["easy"]["loss_val"],
                 "Target Easy - Loss", "Epoch", "Loss", yscale='log')

    # Easy Prediction score
    __plot_trainings(ax[0, 1], metrics["easy"]["acc_train"], metrics["easy"]["acc_val"],
                 "Target Easy - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Medium Loss
    __plot_trainings(ax[1, 0], metrics["medium"]["loss_train"], metrics["medium"]["loss_val"],
                 "Target Medium - Loss", "Epoch", "Loss", yscale='log')

    # Medium Prediction score
    __plot_trainings(ax[1, 1], metrics["medium"]["acc_train"], metrics["medium"]["acc_val"],
                 "Target Medium - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Hard Loss
    __plot_trainings(ax[2, 0], metrics["hard"]["loss_train"], metrics["hard"]["loss_val"],
                 "Target Hard - Loss", "Epoch", "Loss", yscale='log')

    # Hard Prediction score
    __plot_trainings(ax[2, 1], metrics["hard"]["acc_train"], metrics["hard"]["acc_val"],
                    "Target Hard - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Stable Loss
    __plot_trainings(ax[3, 0], metrics["stable"]["loss_train"], metrics["stable"]["loss_val"],
                 "Target Stable - Loss", "Epoch", "Loss", yscale='log')

    # Stable Prediction score
    __plot_trainings(ax[3, 1], metrics["stable"]["acc_train"], metrics["stable"]["acc_val"],
                 "Target Stable - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "trainings_baseline_model.pdf"
    export_figures_to_pdf(pdf_path, fig)


def plot_data_base_toro_vs_zero():

    data_toro = retrieve_log_data(log_path= TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Medium/logs/training_progress.txt")
    data_zero = retrieve_log_data(log_path= TRAINED_MODELS_DIR / "predictors/Baseline_Zero_Medium/logs/training_progress.txt")

    if data_toro["target_type"] != data_zero["target_type"]:
        print("The target types are different")
        return

    losses = {
        "toro": {
            "train": data_toro["train_losses"],
            "val": data_toro["val_losses"]
        },
        "zero": {
            "train": data_zero["train_losses"],
            "val": data_zero["val_losses"]
        }
    }

    pred_scores = {
        "toro": {
            "train": data_toro["train_accuracies"],
            "val": data_toro["val_accuracies"]
        },
        "zero": {
            "train": data_zero["train_accuracies"],
            "val": data_zero["val_accuracies"]
        }
    }

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    plt.suptitle(f"Baseline Model - Toroidal padding vs Zero padding ", fontsize=18, fontweight='bold')

    __plot_toro_zero(ax[0], losses, f"Training on {data_toro['target_type'].capitalize()} - Loss", "Epoch", "Loss", yscale='log')
    __plot_toro_zero(ax[1], pred_scores, f"Training on {data_toro['target_type'].capitalize()} - Prediction score", "Epoch", "Prediction score (%)", legend_loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "toro_zero_baseline_model_medium.pdf"
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

    model_easy_checkpoint   = model_easy   / "checkpoints" / f"predictor_{checkpoint_index}.pt"
    model_medium_checkpoint = model_medium / "checkpoints" / f"predictor_{checkpoint_index}.pt"
    model_hard_checkpoint   = model_hard   / "checkpoints" / f"predictor_{checkpoint_index}.pt"
    model_stable_checkpoint = model_stable / "checkpoints" / f"predictor_{checkpoint_index}.pt"

    avg_easy_p_scores_each_n_cells   = get_prediction_score_n_cells_initial(model_easy_checkpoint)
    avg_medium_p_scores_each_n_cells = get_prediction_score_n_cells_initial(model_medium_checkpoint)
    avg_hard_p_scores_each_n_cells   = get_prediction_score_n_cells_initial(model_hard_checkpoint)
    avg_stable_p_scores_each_n_cells = get_prediction_score_n_cells_initial(model_stable_checkpoint)


    # Save all data to load it in the future
    base_folder = OUTPUTS_DIR / "info_baseline_model_pred_score"
    np.save(base_folder / "avg_easy_scores_each_checkpoint.npy", avg_easy_scores_each_checkpoint)
    np.save(base_folder / "avg_medium_scores_each_checkpoint.npy", avg_medium_scores_each_checkpoint)
    np.save(base_folder / "avg_hard_scores_each_checkpoint.npy", avg_hard_scores_each_checkpoint)
    np.save(base_folder / "avg_stable_scores_each_checkpoint.npy", avg_stable_scores_each_checkpoint)

    np.save(base_folder / "avg_easy_p_score.npy", avg_easy_p_score)
    np.save(base_folder / "avg_medium_p_score.npy", avg_medium_p_score)
    np.save(base_folder / "avg_hard_p_score.npy", avg_hard_p_score)
    np.save(base_folder / "avg_stable_p_score.npy", avg_stable_p_score)

    np.save(base_folder / "avg_easy_p_scores_each_n_cells.npy", avg_easy_p_scores_each_n_cells)
    np.save(base_folder / "avg_medium_p_scores_each_n_cells.npy", avg_medium_p_scores_each_n_cells)
    np.save(base_folder / "avg_hard_p_scores_each_n_cells.npy", avg_hard_p_scores_each_n_cells)
    np.save(base_folder / "avg_stable_p_scores_each_n_cells.npy", avg_stable_p_scores_each_n_cells)

    # Plotting
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    fig.suptitle("Baseline Model - Prediction Score Analysis on Test Dataset", fontsize=18, fontweight='bold')


    # EASY
    axs[0, 0].plot(avg_easy_scores_each_checkpoint,
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
    axs[1, 0].plot(avg_medium_scores_each_checkpoint,
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
    axs[2, 0].plot(avg_hard_scores_each_checkpoint,
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
    axs[3, 0].plot(avg_stable_scores_each_checkpoint,
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

    plot_baseline_pred_score_each_target()

    # train(Predictor_Baseline(TOPOLOGY_TOROIDAL), CONFIG_TARGET_MEDIUM)
    # train(Predictor_ResNet(TOPOLOGY_TOROIDAL, 10, 64), CONFIG_TARGET_MEDIUM)
    # train(Predictor_UNet(), CONFIG_TARGET_MEDIUM)

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


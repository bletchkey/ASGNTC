from pathlib import Path
import sys
import logging
import os
import gc
import re
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
                                           Predictor_UNet, Predictor_ResNetAttention


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
            ax.annotate(f'Min: {min(val_data):.3f}', (min_val_epoch, min(val_data)), textcoords="offset points", xytext=(-5,20),    ha='center',    color=val_color, fontsize=8)
            ax.annotate(f'Max: {max(val_data):.3f}', (max_val_epoch, max(val_data)), textcoords="offset points", xytext=(30,0),    ha='center',   color=val_color, fontsize=8)

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


def plot_baseline_on_all_targets():

    # Retrieve log data for all baseline models
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

    # Extract training and validation losses and prediction scores
    metrics = {
        "easy": {
            "loss_train": data_easy["train_losses"],
            "loss_val": data_easy["val_losses"],
            "pred_score_train": data_easy["train_prediction_scores"],
            "pred_score_val": data_easy["val_prediction_scores"]
        },
        "medium": {
            "loss_train": data_medium["train_losses"],
            "loss_val": data_medium["val_losses"],
            "pred_score_train": data_medium["train_prediction_scores"],
            "pred_score_val": data_medium["val_prediction_scores"]
        },
        "hard": {
            "loss_train": data_hard["train_losses"],
            "loss_val": data_hard["val_losses"],
            "pred_score_train": data_hard["train_prediction_scores"],
            "pred_score_val": data_hard["val_prediction_scores"]
        },
        "stable": {
            "loss_train": data_stable["train_losses"],
            "loss_val": data_stable["val_losses"],
            "pred_score_train": data_stable["train_prediction_scores"],
            "pred_score_val": data_stable["val_prediction_scores"]
        }
    }

    # Plotting
    fig, ax = plt.subplots(4, 2, figsize=(10, 12))
    # plt.suptitle("Baseline Model - Training", fontsize=18, fontweight='bold')

    # Easy Loss
    __plot_trainings(ax[0, 0], metrics["easy"]["loss_train"], metrics["easy"]["loss_val"],
                 "Target Easy - Loss", "Epoch", "Loss", yscale='log')

    # Easy Prediction score
    __plot_trainings(ax[0, 1], metrics["easy"]["pred_score_train"], metrics["easy"]["pred_score_val"],
                 "Target Easy - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Medium Loss
    __plot_trainings(ax[1, 0], metrics["medium"]["loss_train"], metrics["medium"]["loss_val"],
                 "Target Medium - Loss", "Epoch", "Loss", yscale='log')

    # Medium Prediction score
    __plot_trainings(ax[1, 1], metrics["medium"]["pred_score_train"], metrics["medium"]["pred_score_val"],
                 "Target Medium - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Hard Loss
    __plot_trainings(ax[2, 0], metrics["hard"]["loss_train"], metrics["hard"]["loss_val"],
                 "Target Hard - Loss", "Epoch", "Loss", yscale='log')

    # Hard Prediction score
    __plot_trainings(ax[2, 1], metrics["hard"]["pred_score_train"], metrics["hard"]["pred_score_val"],
                    "Target Hard - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Stable Loss
    __plot_trainings(ax[3, 0], metrics["stable"]["loss_train"], metrics["stable"]["loss_val"],
                 "Target Stable - Loss", "Epoch", "Loss", yscale='log')

    # Stable Prediction score
    __plot_trainings(ax[3, 1], metrics["stable"]["pred_score_train"], metrics["stable"]["pred_score_val"],
                 "Target Stable - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "trainings_baseline_model.pdf"
    export_figures_to_pdf(pdf_path, fig)


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
                ax.annotate(f'Min: {min(data):.3f}', (min_val_epoch, min(data)), textcoords="offset points", xytext=(-10,20),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.3f}', (max_val_epoch, max(data)), textcoords="offset points", xytext=(30,0),    ha='center',   color=color, fontsize=8)

                if (data[-1] >= 90):
                    pos_y = -15
                else:
                    pos_y = 8

                ax.annotate(f'Last: {data[-1]:.3f}', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=color, fontsize=8)

            else:
                ax.annotate(f'Min: {min(data):.1f}%', (min_val_epoch, min(data)), textcoords="offset points", xytext=(15,20),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.1f}%', (max_val_epoch, max(data)), textcoords="offset points", xytext=(5,-25),    ha='center',   color=color, fontsize=8)

                if (data[-1] >= 90):
                    pos_y = -15
                else:
                    pos_y = 8

                ax.annotate(f'Last: {data[-1]:.1f}%', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-20, pos_y),   ha='center', color=color, fontsize=8)


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
            "train": data_toro["train_prediction_scores"],
            "val": data_toro["val_prediction_scores"]
        },
        "zero": {
            "train": data_zero["train_prediction_scores"],
            "val": data_zero["val_prediction_scores"]
        }
    }

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    # plt.suptitle(f"Baseline Model - Toroidal padding vs Zero padding ", fontsize=18, fontweight='bold')

    __plot_toro_zero(ax[0], losses, f"Training Loss - Target {data_toro['target_type'].capitalize()} ", "Epoch", "Loss", yscale='log')
    __plot_toro_zero(ax[1], pred_scores, f"Training Prediction score - Target {data_toro['target_type'].capitalize()}", "Epoch", "Prediction score (%)", legend_loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "toro_zero_baseline_model_medium.pdf"
    export_figures_to_pdf(pdf_path, fig)


def __plot_pred_score_test_set(ax, scores, title, xlabel, ylabel):
    ax.plot(scores, color='#385BA8', linestyle='-', linewidth=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 100])


def __plot_performance_by_cells(ax, scores, avg_score, title, xlabel, ylabel):
    ax.scatter(np.arange(0, GRID_SIZE**2+1), scores, color='#6e0b2f', label='Prediction score for each number of initial cells', marker='x', s=2)
    ax.hlines(avg_score, 0, GRID_SIZE**2, colors='#75849c', linestyles="--", label='Overall average prediction score', linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, GRID_SIZE**2])
    ax.set_ylim([0, 100])

    specific_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1024]
    ax.set_xticks(specific_ticks)  # Apply the specific tick marks to the x-axis


def plot_baseline_pred_score_analysis():

    targets = ['easy', 'medium', 'hard', 'stable']

    model_folder_path = TRAINED_MODELS_DIR / "predictors"

    model_easy   = model_folder_path / "Baseline_Toroidal_Easy"
    model_medium = model_folder_path / "Baseline_Toroidal_Medium"
    model_hard   = model_folder_path / "Baseline_Toroidal_Hard"
    model_stable = model_folder_path / "Baseline_Toroidal_Stable"

    pred_scores_each_checkpoint ={
        'easy'  : [],
        'medium': [],
        'hard'  : [],
        'stable': []
    }


    log = (OUTPUTS_DIR / "test_set_pred_score.txt").read_text()

    if not log:
        pred_scores_each_checkpoint["easy"]   = get_prediction_score(model_easy),
        pred_scores_each_checkpoint["medium"] = get_prediction_score(model_medium),
        pred_scores_each_checkpoint["hard"]   = get_prediction_score(model_hard),
        pred_scores_each_checkpoint["stable"] = get_prediction_score(model_stable)

        # Multiply each score by 100 to convert to percentage
        pred_scores_each_checkpoint = {key: [score * 100 for score in scores] for key, scores in pred_scores_each_checkpoint.items()}

    else:
        # Regular expression to match lines with scores
        pattern = r"Checkpoint (\w+) predictor_\d+ - Prediction score: ([\d.]+)%"

        # Process each line that matches the pattern
        for match in re.finditer(pattern, log):
            category = match.group(1).lower()
            score = float(match.group(2))

            if category == 'easy':
                pred_scores_each_checkpoint['easy'].append(score)
            elif category == 'medium':
                pred_scores_each_checkpoint['medium'].append(score)
            elif category == 'hard':
                pred_scores_each_checkpoint['hard'].append(score)
            elif category == 'stable':
                pred_scores_each_checkpoint['stable'].append(score)

    checkpoint_index = 100
    pred_score_checkpoint = {
        'easy'  : pred_scores_each_checkpoint['easy'][checkpoint_index-1],
        'medium': pred_scores_each_checkpoint['medium'][checkpoint_index-1],
        'hard'  : pred_scores_each_checkpoint['hard'][checkpoint_index-1],
        'stable': pred_scores_each_checkpoint['stable'][checkpoint_index-1]
    }

    model_easy_checkpoint   = model_easy   / "checkpoints" / f"predictor_{checkpoint_index}.pth.tar"
    model_medium_checkpoint = model_medium / "checkpoints" / f"predictor_{checkpoint_index}.pth.tar"
    model_hard_checkpoint   = model_hard   / "checkpoints" / f"predictor_{checkpoint_index}.pth.tar"
    model_stable_checkpoint = model_stable / "checkpoints" / f"predictor_{checkpoint_index}.pth.tar"

    pred_scores_each_n_cells = {
        'easy'  : get_prediction_score_n_cells_initial(model_easy_checkpoint),
        'medium': get_prediction_score_n_cells_initial(model_medium_checkpoint),
        'hard'  : get_prediction_score_n_cells_initial(model_hard_checkpoint),
        'stable': get_prediction_score_n_cells_initial(model_stable_checkpoint)
    }

    # Multiply each score by 100 to convert to percentage
    pred_scores_each_n_cells = {key: [score * 100 for score in scores] for key, scores in pred_scores_each_n_cells.items()}

    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    # fig.suptitle("Baseline Model - Prediction Score Analysis on Test Dataset", fontsize=18, fontweight='bold')

    for i, target in enumerate(targets):

        __plot_pred_score_test_set(axs[i, 0], pred_scores_each_checkpoint[target],
                                   f"Target {target.capitalize()} - Prediction score", "Epoch", "Prediction score (%)")

        __plot_performance_by_cells(axs[i, 1],
                                    pred_scores_each_n_cells[target],
                                    pred_score_checkpoint[target],
                                    f"Target {target.capitalize()} - Prediction score - Epoch {checkpoint_index}", "Number of initial   cells", "Prediction score (%)")

    plt.tight_layout()
    pdf_path = OUTPUTS_DIR / "baseline_pred_score_analysis.pdf"
    export_figures_to_pdf(pdf_path, fig)


def __plot_base_unet(ax, data_base, data_unet, title, xlabel, ylabel, yscale='linear', ylim=None, legend_loc='upper right'):
    # Use distinct, easily distinguishable colors
        base_color = 'darkblue'
        unet_color = '#fc7f03'

        # Plot data with markers and different line styles
        ax.plot(data_base, label='Baseline', color=base_color, marker='o', markersize=1.5)
        ax.plot(data_unet, label='UNet', color=unet_color, linestyle="--", linewidth=0.5, marker='x', markersize=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)

        # Automatically adjust ylim based on data if not provided
        if ylim is None:
            all_data = data_base + data_unet
            buffer = (max(all_data) - min(all_data)) * 0.8
            ylim = [min(all_data) - buffer, max(all_data) + buffer]
        ax.set_ylim(ylim)
        print(ylim)

        ax.legend(loc=legend_loc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Annotate important points
        for data in [data_base, data_unet]:

            color = base_color if data == data_base else unet_color
            min_val_epoch  = data.index(min(data))
            max_val_epoch  = data.index(max(data))
            last_val_epoch = len(data) - 1

            if yscale == 'log':
                ax.annotate(f'Min: {min(data):.3f}', (min_val_epoch, min(data)), textcoords="offset points", xytext=(-10,2),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.3f}', (max_val_epoch, max(data)), textcoords="offset points", xytext=(30,0),    ha='center',   color=color, fontsize=8)

                ax.annotate(f'Last: {data[-1]:.3f}', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-20, -10),   ha='center', color=color, fontsize=8)

            else:
                ax.annotate(f'Min: {min(data):.1f}%', (min_val_epoch, min(data)), textcoords="offset points", xytext=(15,20),    ha='center',    color=color, fontsize=8)
                ax.annotate(f'Max: {max(data):.1f}%', (max_val_epoch, max(data)), textcoords="offset points", xytext=(5,-10),    ha='center',   color=color, fontsize=8)

                ax.annotate(f'Last: {data[-1]:.1f}%', (last_val_epoch, data[-1]), textcoords="offset points", xytext=(-10, -15),   ha='center', color=color, fontsize=8)


def plot_baseline_vs_unet():
    # Retrieve log data
    log_paths = {
        "base_easy"  : TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Easy/logs/training_progress.txt",
        "base_medium": TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Medium/logs/training_progress.txt",
        "base_hard"  : TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Hard/logs/training_progress.txt",
        "base_stable": TRAINED_MODELS_DIR / "predictors/Baseline_Toroidal_Stable/logs/training_progress.txt",
        "unet_easy"  : TRAINED_MODELS_DIR / "predictors/UNet_Toroidal_Easy/logs/training_progress.txt",
        "unet_medium": TRAINED_MODELS_DIR / "predictors/UNet_Toroidal_Medium/logs/training_progress.txt",
        "unet_hard"  : TRAINED_MODELS_DIR / "predictors/UNet_Toroidal_Hard/logs/training_progress.txt",
        "unet_stable": TRAINED_MODELS_DIR / "predictors/UNet_Toroidal_Stable/logs/training_progress.txt"
    }

    data = {key: retrieve_log_data(log_paths[key]) for key in log_paths.keys()}

    stats = { key: {"loss_val": data[key]["val_losses"],
                    "pred_score_val": data[key]["val_prediction_scores"]}
                    for key in data.keys() }

    # Plotting
    fig, ax = plt.subplots(4, 2, figsize=(10, 12))
    # plt.suptitle("Baseline Model vs UNet Model - Training", fontsize=18, fontweight='bold')

    # Easy Loss
    __plot_base_unet(ax[0, 0], stats["base_easy"]["loss_val"], stats["unet_easy"]["loss_val"],
                 "Target Easy - Loss", "Epoch", "Loss", yscale='log')

    # Easy Prediction score
    __plot_base_unet(ax[0, 1], stats["base_easy"]["pred_score_val"], stats["unet_easy"]["pred_score_val"],
                    "Target Easy - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Medium Loss
    __plot_base_unet(ax[1, 0], stats["base_medium"]["loss_val"], stats["unet_medium"]["loss_val"],
                 "Target Medium - Loss", "Epoch", "Loss", yscale='log')

    # Medium Prediction score
    __plot_base_unet(ax[1, 1], stats["base_medium"]["pred_score_val"], stats["unet_medium"]["pred_score_val"],
                    "Target Medium - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Hard Loss
    __plot_base_unet(ax[2, 0], stats["base_hard"]["loss_val"], stats["unet_hard"]["loss_val"],
                 "Target Hard - Loss", "Epoch", "Loss", yscale='log')

    # Hard Prediction score
    __plot_base_unet(ax[2, 1], stats["base_hard"]["pred_score_val"], stats["unet_hard"]["pred_score_val"],
                    "Target Hard - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100], legend_loc='lower right')

    # Stable Loss
    __plot_base_unet(ax[3, 0], stats["base_stable"]["loss_val"], stats["unet_stable"]["loss_val"],
                 "Target Stable - Loss", "Epoch", "Loss", yscale='log')

    # Stable Prediction score
    __plot_base_unet(ax[3, 1], stats["base_stable"]["pred_score_val"], stats["unet_stable"]["pred_score_val"],
                    "Target Stable - Prediction score", "Epoch", "Prediction score (%)", ylim=[0, 100])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    pdf_path = OUTPUTS_DIR / "baseline_vs_unet_model.pdf"
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
    # plot_baseline_pred_score_analysis()
    # plot_baseline_vs_unet()

    # train(Predictor_ResNet(TOPOLOGY_TOROIDAL, 10, 64), CONFIG_TARGET_MEDIUM)

    train(Predictor_ResNetAttention(128), CONFIG_TARGET_EASY)

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


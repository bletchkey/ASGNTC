import re
from pathlib import Path
import logging
import os
import torch

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from configs.constants import *


def get_elapsed_time_str(times: list) -> str:
    """
    Function to get the elapsed time in an easily readable format
    All the times are summed together and then converted to hours, minutes and seconds

    Args:
        times (list): The list of times in seconds

    Returns:
        time_format (str): The elapsed time in the format "Hh Mm Ss"

    """
    seconds = sum(times) if isinstance(times, list) else times
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60  # Remaining minutes after converting to hours
    remaining_seconds = int(seconds % 60)

    # Format time
    time_format = f"{hours}h {remaining_minutes}m {remaining_seconds}s"

    return time_format


def retrieve_log_data(log_path: str):

    log = Path(log_path).read_text()

    # Regular expression patterns for extracting data
    epoch_pattern      = re.compile(r"Epoch: (\d+)/\d+")
    train_loss_pattern = re.compile(r"Losses P \[train: ([0-9.]+), val: ([0-9.]+)\]")
    train_acc_pattern  = re.compile(r"Accuracies P \[train: ([0-9.]+)%, val: ([0-9.]+)%\]")
    target_type        = re.compile(r"Prediction target configuration type: (\w+)")
    input_type         = re.compile(r"Prediction input configuration type: (\w+)")

    # Lists to store extracted data
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Iterate through each line in the log
    for line in log.split('\n'):
        epoch_match      = epoch_pattern.search(line)
        train_loss_match = train_loss_pattern.search(line)
        train_acc_match  = train_acc_pattern.search(line)

        if epoch_match and train_loss_match and train_acc_match:
            epochs.append(int(epoch_match.group(1)))
            train_losses.append(float(train_loss_match.group(1)))
            val_losses.append(float(train_loss_match.group(2)))
            train_accuracies.append(float(train_acc_match.group(1)))
            val_accuracies.append(float(train_acc_match.group(2)))

    # Extract target type
    target_type_match = target_type.search(log)
    target_type = target_type_match.group(1) if target_type_match else "unknown"

    # Extract input type
    input_type_match = input_type.search(log)
    input_type = input_type_match.group(1) if input_type_match else "unknown"

    training_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "target_type": target_type,
        "input_type": input_type
    }

    return training_data


def export_figures_to_pdf(pdf_path, figs):
    """
    Saves one or multiple matplotlib figures to a single PDF file.

    Parameters:
        pdf_path (str): The path to the PDF file where figures will be saved.
        figs (matplotlib.figure.Figure or list of matplotlib.figure.Figure):
            A single matplotlib figure or a list of figures to be saved.

    """
    if not isinstance(figs, list):
        figs = [figs]

    try:
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05, dpi=300)
                plt.close(fig)
        logging.debug(f"Saved {len(figs)} figure(s) to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to save figure(s) to {pdf_path}: {str(e)}")


def get_latest_checkpoint_path(model_folder_path: str) -> str:
    """
    Function to get the latest checkpoint file in a directory

    Args:
        model_folder_path (str): The path to the model folder

    Returns:
        latest_checkpoint (str): The path to the latest checkpoint file

    """

    model_checkpoints = sorted(os.listdir(model_folder_path / "checkpoints"))

    latest_checkpoint = model_folder_path / "checkpoints" / model_checkpoints[-1]

    return latest_checkpoint


def get_model_data_from_checkpoint(checkpoint_path: Path) -> dict:

    if not checkpoint_path.exists():
        logging.error("get_model_infos_from_checkpoint: Checkpoint not found")
        return

    checkpoint = torch.load(checkpoint_path)

    data = {
        CHECKPOINT_MODEL_STATE_DICT_KEY       : checkpoint[CHECKPOINT_MODEL_STATE_DICT_KEY],
        CHECKPOINT_MODEL_OPTIMIZER_STATE_DICT : checkpoint[CHECKPOINT_MODEL_OPTIMIZER_STATE_DICT],
        CHECKPOINT_MODEL_ARCHITECTURE_KEY     : checkpoint[CHECKPOINT_MODEL_ARCHITECTURE_KEY],
        CHECKPOINT_MODEL_TYPE_KEY             : checkpoint[CHECKPOINT_MODEL_TYPE_KEY],
        CHECKPOINT_MODEL_NAME_KEY             : checkpoint[CHECKPOINT_MODEL_NAME_KEY],
        CHECKPOINT_EPOCH_KEY                  : checkpoint[CHECKPOINT_EPOCH_KEY],
        CHECKPOINT_TRAIN_LOSS_KEY             : checkpoint[CHECKPOINT_TRAIN_LOSS_KEY],
        CHECKPOINT_VAL_LOSS_KEY               : checkpoint[CHECKPOINT_VAL_LOSS_KEY],
        CHECKPOINT_SEED_KEY                   : checkpoint[CHECKPOINT_SEED_KEY],
        CHECKPOINT_DATE_KEY                   : checkpoint[CHECKPOINT_DATE_KEY],
        CHECKPOINT_N_TIMES_TRAINED_KEY        : checkpoint[CHECKPOINT_N_TIMES_TRAINED_KEY],
        CHECKPOINT_P_INPUT_TYPE               : checkpoint[CHECKPOINT_P_INPUT_TYPE],
        CHECKPOINT_P_TARGET_TYPE              : checkpoint[CHECKPOINT_P_TARGET_TYPE]
    }

    return data


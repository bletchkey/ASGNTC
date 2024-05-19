import re
from pathlib import Path

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


import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import re

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.paths import TRAININGS_DIR

from src.gol_adv_sys.utils.helper_functions import save_loss_acc_plot

"""
This script may not work anymore, as the log file format may have changed.

"""

def main():

    # Initialize lists to hold the loss values
    loss_train = []
    loss_val = []
    lr_values = []

    # Path to your log file
    training_path = TRAININGS_DIR / "2024-03-12_06-52-54"
    file_path = "logs/asgntc.txt"
    final_path = training_path / file_path

    pattern_decimal = r"-?\d+\.?\d*"

    # Read the file and extract loss values
    with open(final_path, 'r') as file:
        for line in file:
            # Check if line contains loss information
            if "Loss P (train):" in line and "Loss P (val):" in line:
                # Split line by commas and extract relevant parts
                parts = line.split(',')
                train_loss_str = parts[1].split(': ')[1]
                val_loss_str = parts[2].split(': ')[1]
                lr_str = parts[3].split(': ')[1]

                match = re.search(pattern_decimal, lr_str)
                if match:
                    number_part = match.group()
                    lr_values.append(float(number_part))

                loss_train.append(float(train_loss_str))
                loss_val.append(float(val_loss_str))

    # Save the plot
    acc_train = None
    acc_val = None
    save_loss_acc_plot(loss_train, loss_val, acc_train, acc_val, lr_values, training_path)

    return 0


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


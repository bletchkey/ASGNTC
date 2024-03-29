import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from pathlib import Path

from configs.constants import *


def get_config_from_batch(batch: torch.Tensor, type: str, device: torch.device) -> torch.Tensor:
    """
    Function to get a batch of a certain type of configuration from the batch itself

    Args:
        batch (torch.Tensor): The batch containing the configurations
        type (str): The type of configuration to retrieve
        device (torch.device): The device to use for computation

    Returns:
        torch.Tensor: The configuration specified by the type

    """
    # Ensure the batch has the expected dimensions (5D tensor)
    if batch.dim() != 5:
        raise RuntimeError(f"Expected batch to have 5 dimensions, got {batch.dim()}")

    # Mapping from type to index in the batch
    config_indices = {
        CONFIG_INITIAL: 0,
        CONFIG_FINAL: 1,
        CONFIG_METRIC_EASY: 2,
        CONFIG_METRIC_MEDIUM: 3,
        CONFIG_METRIC_HARD: 4,
        CONFIG_METRIC_STABLE: 5,
    }

    # Validate and retrieve the configuration index
    if type not in config_indices:
        raise ValueError(f"Invalid type: {type}. Valid types are {list(config_indices.keys())}")

    config_index = config_indices[type]

    # Extract and return the configuration
    return batch[:, config_index, :, :, :].to(device)


def test_predictor_model_dataset(test_set: torch.utils.data.DataLoader,
                                 config_type_pred_input: str,
                                 target_config: str, model_p: torch.nn.Module, device: torch.device) -> dict:
    """
    Function to test the predictor model.

    Args:
        test_set (torch.utils.data.DataLoader): The test set containing data and metadata
        target_config (str): The type of configuration to predict (tipically the metric configuration)
        model_p (torch.nn.Module): The predictor model
        device (torch.device): The device used for computation

    Returns:
        dict: The dictionary containing the test results.

    """

    model_p.eval()  # Set the model to evaluation mode

    batch_data = None
    batch_metadata_aggregated = {}
    all_predictions = None

    with torch.no_grad():
        for batch, batch_metadata in test_set:

            # Get prediction for the current batch
            prediction = model_p(get_config_from_batch(batch, config_type_pred_input, device))

            # Aggregate batch data if already exists, else initialize
            batch_data = torch.cat((batch_data, batch), dim=0) if batch_data is not None else batch

            # Aggregate predictions if already exists, else initialize
            all_predictions = torch.cat((all_predictions, prediction), dim=0) if all_predictions is not None else prediction

            # Aggregate batch metadata
            for key in batch_metadata.keys():
                if key in batch_metadata_aggregated:
                    batch_metadata_aggregated[key] = torch.cat((batch_metadata_aggregated[key], batch_metadata[key]), dim=0)
                else:
                    batch_metadata_aggregated[key] = batch_metadata[key]

    # Construct metadata from aggregated data
    metadata = {
        META_ID: batch_metadata_aggregated[META_ID],
        META_N_CELLS_INITIAL : batch_metadata_aggregated[META_N_CELLS_INITIAL ],
        META_N_CELLS_FINAL: batch_metadata_aggregated[META_N_CELLS_FINAL],
        META_TRANSIENT_PHASE: batch_metadata_aggregated[META_TRANSIENT_PHASE],
        META_PERIOD: batch_metadata_aggregated[META_PERIOD],
        "input_config" : config_type_pred_input,
        "target_config": target_config,
    }

    # Assuming __create_data_dict is a function that prepares your data dictionary
    data = __create_data_dict(batch_data, all_predictions, metadata)

    return data


def save_progress_plot_dataset(plot_data: dict, epoch: int, results_path: str) -> None:
    """
    Function to save the progress plot

    Args:
        plot_data (dict): The dictionary containing the data to plot
        epoch (int): The current epoch
        results_path (str): The path to where the results will be saved

    """

    vmin = 0
    vmax = 1

    titles = [key for key in plot_data.keys()]

    # IDs
    ids = [8154, 87646, 96922, 115472, 179702, 248104]
    id_tensor = plot_data["metadata"]["id"]

    # Check and ensure id_tensor is a PyTorch tensor
    if not isinstance(id_tensor, torch.Tensor):
        id_tensor = torch.tensor(id_tensor)

    indices = []
    for id in ids:
        # Find where the ID matches in the tensor
        matched_indices = torch.where(id_tensor == id)[0]

        # Check if there is at least one match and get the first occurrence
        if matched_indices.nelement() > 0:
            index = matched_indices[0].item()
            indices.append(index)
        else:
            print(f"ID {id} not found.")
            indices.append(None)

    n_rows = len(indices)
    n_cols = len(titles)

    current_epoch = epoch + 1

    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))

    plt.suptitle(f"Epoch {current_epoch}", fontsize=32)

    # Convert to NumPy
    for key in plot_data.keys():
        if key == "metadata":
            continue
        plot_data[key] = plot_data[key].detach().cpu().numpy().squeeze()

    # Plot each data in a subplot
    for i in range(n_rows):
        for j, key in enumerate(titles):
            if key != "metadata":
                img_data = plot_data[key][indices[i]]
                axs[i, j].imshow(img_data, cmap='gray', vmin=vmin, vmax=vmax)
                axs[i, j].patch.set_edgecolor('black')
                axs[i, j].patch.set_linewidth(1)
                if key == "input" and i == 0:
                    axs[i, j].set_title(f"input - {plot_data['metadata']['input_config']}", pad=6, fontsize=24)
                elif key == "target" and i == 0:
                    axs[i, j].set_title(f"target - {plot_data['metadata']['target_config']}", pad=6, fontsize=24)
                elif i == 0:
                    axs[i, j].set_title(titles[j], pad=6, fontsize=24)
                else:
                    axs[i, j].set_title("", pad=6)
            else:
                text_data = ""
                for k in plot_data[key].keys():
                    if k == "target_config" or k == "input_config":
                        continue
                    text_data += f"{k}: {plot_data[key][k][indices[i]]}\n"
                axs[i, j].text(0.1, 0.5, text_data, fontsize=18, ha='left', va='center', transform=axs[i, j].transAxes)

            axs[i, j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(Path(results_path, f"epoch_{current_epoch}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_loss_acc_plot(losses_p_train: list, losses_p_val: list,
                       accuracies_p_train: list, accuracies_p_val: list,
                       learning_rates: list, path: str) -> None:
    """
    Function to save the losses plot

    Args:
        losses_p_train (list): The training losses for the predictor model
        losses_p_val (list): The validation losses for the predictor model
        accuracies_p_train (list): The training accuracies for the predictor model
        accuracies_p_val (list): The validation accuracies for the predictor model
        learning_rates (list): The learning rates used during training for each epoch
        path (str): The path to where the results will be saved

    """

    epochs = list(range(1, len(losses_p_train) + 1))

    # Detecting change indices more robustly
    if len(learning_rates) > 1:
        change_indices = [i for i in range(1, len(learning_rates)) if learning_rates[i] != learning_rates[i-1]]
    else:
        change_indices = []

    change_epochs = [epochs[i] for i in change_indices]

    # Get the learning rate values at change points
    change_lr_values = [learning_rates[i] for i in change_indices]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training and validation losses
    ax[0].plot(epochs, losses_p_train, label="Training Loss", color='blue', linewidth=0.7)
    ax[0].plot(epochs, losses_p_val, label="Validation Loss", color='orange', linewidth=0.7, linestyle='--')

    ax[0].set_yscale('log')
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].legend(loc='upper right')

    # Plot training and validation accuracies
    ax[1].plot(epochs, accuracies_p_train, label="Training Accuracy", color='blue', linewidth=0.7)
    ax[1].plot(epochs, accuracies_p_val, label="Validation Accuracy", color='orange', linewidth=0.7, linestyle='--')

    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Accuracy", fontsize=12)
    ax[1].legend(loc='lower right')

    # Mark learning rate changes on the x-axis and annotate the learning rate value
    ymin, ymax = ax[0].get_ylim()
    for epoch, lr_value in zip(change_epochs, change_lr_values):
        ax[0].plot([epoch, epoch], [ymin, ymax], color='green', linestyle='-', linewidth=0.1)  # Vertical line for each LR change
        ax[0].annotate(f'{lr_value:.2e}',  # Formatting the learning rate value
                     (epoch, ymin),  # Position at the bottom of the plot
                     textcoords="offset points", xytext=(-10,0), ha='left', fontsize=2, color='green')

    plt.tight_layout()
    plt.savefig(Path(path, "loss_acc_graph.png"), dpi=400)
    plt.close()


def __create_data_dict(batches: torch.Tensor, prediction: torch.Tensor, metadata: dict) -> dict:
    """
    Function to create the dictionary to plot the data

    Args:
        data (dict): The dictionary containing the data

    Returns:
        dict: The dictionary containing the data

    """

    device = batches.device

    initial = get_config_from_batch(batches, CONFIG_INITIAL, device)
    final   = get_config_from_batch(batches, CONFIG_FINAL, device)
    input   = get_config_from_batch(batches, metadata["input_config"], device)
    target  = get_config_from_batch(batches, metadata["target_config"], device)

    data = {
        "initial": initial,
        "final": final,
        "metadata": metadata,
        "input": input,
        "target": target,
        "predicted": prediction,
    }

    return data


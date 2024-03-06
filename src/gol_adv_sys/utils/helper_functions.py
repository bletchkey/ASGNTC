import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_config


"""
Function to get the dataloader for the training of the predictor model

Add configurations to the dataloader until it reaches the maximum number of configurations
If max number of configurations is reached, remove the oldest configurations to make room for the new ones

This method implements a sliding window approach

"""
def get_data_tensor(data_tensor, model_g, topology, init_config_type, device):

    new_configs = generate_new_batches(model_g, constants.n_batches, topology, init_config_type, device)

    # If data_tensor is None, initialize it with new_configs
    if data_tensor is None:
        data_tensor = new_configs
    else:
        # Concatenate current data with new_configs
        combined_tensor = torch.cat([data_tensor, new_configs], dim=0)

        # If the combined size exceeds the max allowed size, trim the oldest entries
        max_size = constants.n_max_batches * constants.bs
        if combined_tensor.size(0) > max_size:
            # Calculate number of entries to drop from the start to fit the new_configs
            excess_entries = combined_tensor.size(0) - max_size
            data_tensor = combined_tensor[excess_entries:]
        else:
            data_tensor = combined_tensor

    print(f"Data tensor shape: {data_tensor.shape}, used for creating the dataloader.")

    return data_tensor


"""
Function to generate new batches of configurations

"""
def generate_new_batches(model_g, n_batches, topology, init_config_type, device):

    configs = []

    for _ in range(n_batches):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)
        generated_config = model_g(noise)
        initial_config = get_init_config(generated_config, init_config_type)

        with torch.no_grad():
            simulated_config, simulated_metrics = simulate_config(config=initial_config, topology=topology,
                                                                  steps=constants.n_simulation_steps, calculate_final_config=False,
                                                                  device=device)
        configs.append({
            "initial": initial_config,
            "simulated":  simulated_config,
            "metric_easy": simulated_metrics["easy"],
            "metric_medium": simulated_metrics["medium"],
            "metric_hard": simulated_metrics["hard"],
        })

    # Create a tensor from the list of configurations
    concatenated_tensors = []
    keys = configs[0].keys()
    for key in keys:
        concatenated_tensor = torch.cat([config[key] for config in configs], dim=0)
        concatenated_tensors.append(concatenated_tensor)
    # Stack the tensors
    data = torch.stack(concatenated_tensors, dim=1)
    # Shuffle the data
    data = data[torch.randperm(data.size(0))]

    return data


"""
Function to test the models

"""
def test_models(model_g, model_p, topology, init_config_type, fixed_noise, metric_type, device):
    data = {
        "generated": None,
        "initial": None,
        "simulated": None,
        "metric": None,
        "predicted_metric": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_config_fixed = model_g(fixed_noise)
        data["generated"] = generated_config_fixed
        data["initial"]   = get_init_config(generated_config_fixed, init_config_type)
        data["simulated"], sim_metrics = simulate_config(config=data["initial"], topology=topology,
                                                         steps=constants.n_simulation_steps, calculate_final_config=False,
                                                         device=device)
        data["metric"] = sim_metrics[metric_type]
        data["predicted_metric"] = model_p(data["initial"])

    return data


"""
Function to test the predictor model

"""
def test_predictor_model(test_set, metric_type, model_p, device):

    # Create an iterator from the data_loader
    data_iterator = iter(test_set)
    # Fetch the first batch
    batch = next(data_iterator)

    data = {
        "initial": None,
        "final": None,
        "metric": None,
        "predicted_metric": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_p.eval()
        data["predicted_metric"] = model_p(get_config_from_batch(batch, "initial", device))

    data["initial"] = get_config_from_batch(batch, "initial", device)
    data["final"]   = get_config_from_batch(batch, "final", device)
    data["metric"]  = get_config_from_batch(batch, metric_type, device)

    return data

"""
Function to save the progress plot

"""
def save_progress_plot(plot_data, epoch, results_path):
    vmin = 0
    vmax = 1

    titles = []
    # Convert to NumPy
    for key in plot_data.keys():
        plot_data[key] = plot_data[key].detach().cpu().numpy().squeeze()
        titles.append(key)

    current_epoch = epoch+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, constants.bs-1, 4).astype(int)

    # Create figure and subplots
    fig, axs = plt.subplots(len(indices), len(plot_data), figsize=(len(indices)*len(plot_data), len(indices)*4))

    plt.suptitle(f"Epoch {current_epoch}", fontsize=32)

    # Plot each data in a subplot
    for i in range(len(indices)):
        for j, key in enumerate(plot_data.keys()):
            axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray', vmin=vmin, vmax=vmax)
            axs[i, j].set_title(titles[j])

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"epoch_{current_epoch}.png"))
    plt.close(fig)


"""
Function to save the losses plot

"""
def save_losses_plot(losses_p_train, losses_p_val, learning_rates, path):
    epochs = list(range(1, len(losses_p_train) + 1))

    # Detecting change indices more robustly
    if len(learning_rates) > 1:
        change_indices = [i for i in range(1, len(learning_rates)) if learning_rates[i] != learning_rates[i-1]]
    else:
        change_indices = []

    change_epochs = [epochs[i] for i in change_indices]
    change_lr_values = [learning_rates[i] for i in change_indices]  # Get the learning rate values at change points

    fig, ax1 = plt.subplots()

    # Plot training and validation losses
    ax1.plot(epochs, losses_p_train, label="Training Loss", color='blue', linewidth=0.7, linestyle='-')
    ax1.plot(epochs, losses_p_val, label="Validation Loss", color='orange', linewidth=0.7, linestyle='--')

    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Losses", fontsize=12)
    ax1.legend(loc='upper right')

    # Mark learning rate changes on the x-axis and annotate the learning rate value
    ymin, ymax = ax1.get_ylim()
    for epoch, lr_value in zip(change_epochs, change_lr_values):
        ax1.plot([epoch, epoch], [ymin, ymax], color='green', linestyle='-', linewidth=0.1)  # Vertical line for each LR change
        ax1.annotate(f'{lr_value:.2e}',  # Formatting the learning rate value
                     (epoch, ymin),  # Position at the bottom of the plot
                     textcoords="offset points", xytext=(-10,0), ha='left', fontsize=2, color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(path, "losses_graph.png"), dpi=600)
    plt.close()


"""
Function to get the elapsed time in an easily readable format

Args:
    times: list of times in seconds

All the times are summed together and then converted to hours, minutes and seconds

"""
def get_elapsed_time_str(times):

    seconds = sum(times) if isinstance(times, list) else times
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60  # Remaining minutes after converting to hours
    remaining_seconds = int(seconds % 60)

    # Format time
    time_format = f"{hours}h {remaining_minutes}m {remaining_seconds}s"

    return time_format


"""
Get the initial configuration from the generated configuration

"""
def get_init_config(config, init_config_type):

    if init_config_type == constants.INIT_CONFIG_TYPE["threshold"]:
        return __get_init_config_threshold(config)
    elif init_config_type == constants.INIT_CONFIG_TYPE["n_living_cells"]:
        return __get_init_config_n_living_cells(config)
    else:
        raise ValueError(f"Invalid init configuration type: {init_config_type}")


"""
Function to get a batch of a certain type of configuration from the batch itself

"""
def get_config_from_batch(batch, type, device):

    # Ensure the batch has the expected dimensions (5D tensor)
    if batch.dim() != 5:
        raise RuntimeError(f"Expected batch to have 5 dimensions, got {batch.dim()}")

    # Mapping from type to index in the batch
    config_indices = {
        constants.CONFIG_NAMES["initial"]: 0,
        constants.CONFIG_NAMES["final"]: 1,
        constants.CONFIG_NAMES["metric_easy"]: 2,
        constants.CONFIG_NAMES["metric_medium"]: 3,
        constants.CONFIG_NAMES["metric_hard"]: 4
    }

    # Validate and retrieve the configuration index
    if type not in config_indices:
        raise ValueError(f"Invalid type: {type}. Valid types are {list(config_indices.keys())}")

    config_index = config_indices[type]

    # Extract and return the configuration
    return batch[:, config_index, :, :, :].to(device)


"""
Function for adding toroidal padding

"""
def add_toroidal_padding(x):

    if x.dim() != 4:
        raise RuntimeError(f"Expected 4D tensor, got {x.dim()}")

    x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
    x = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)

    return x


"""
From the configuration, get the indices of the n_living_cells highest values and set the rest to 0
The n_living_cells highest values are set to 1

"""
def __get_init_config_n_living_cells(config):

    batch_size, channels, height, width = config.size()
    config_flat = config.view(batch_size, -1)  # Flatten each image in the batch

    # Find the indices of the top values for each image in the batch
    _, indices = torch.topk(config_flat, constants.n_living_cells, dim=1)

    # Create a zero tensor of the same shape
    updated_config = torch.zeros_like(config_flat)

    # Set the top indices to 1 for each image
    updated_config.scatter_(1, indices, 1)

    # Reshape back to the original shape
    updated_config = updated_config.view(batch_size, channels, height, width)

    return updated_config


"""
For every value in configuration, if value is < threshold, set to 0, else set to 1

"""
def __get_init_config_threshold(config):

    return (config > constants.threshold_cell_value).float()


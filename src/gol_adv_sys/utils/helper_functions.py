import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_config, simulate_config_fixed_dataset


"""
Function to get the dataloader for the training of the predictor model

Add configurations to the dataloader until it reaches the maximum number of configurations
If max number of configurations is reached, remove the oldest configurations to make room for the new ones

This method implements a sliding window approach

"""
def get_dataloader(dataloader, model_g, topology, init_config_type, device):

    new_configs = generate_new_batches(model_g, constants.n_batches, topology, init_config_type, device)

    if len(dataloader) + len(new_configs) > constants.n_max_batches:
        dataloader = dataloader[constants.n_batches:]

    dataloader += new_configs

    return dataloader


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
            simulated_config, simulated_metrics = simulate_config(initial_config, topology,
                                                                constants.n_simulation_steps, device)

        configs.append({
            "generated": generated_config,
            "initial": initial_config,
            "simulated":  simulated_config,
            "metric_easy": simulated_metrics["easy"],
            "metric_medium": simulated_metrics["medium"],
            "metric_hard": simulated_metrics["hard"],
        })

    return configs


"""
Function to generate new batches of configurations for fixed dataset

"""
def generate_new_batches_fixed_dataset(model_g, n_configs, batch_size, nz, topology, init_config_type, metric_steps, device):

    configs = []

    n_batches = n_configs // batch_size

    for _ in range(n_batches):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        generated_config = model_g(noise)
        initial_config = get_init_config(generated_config, init_config_type)

        with torch.no_grad():
            final_config, metrics = simulate_config_fixed_dataset(initial_config, topology, metric_steps, device)

        configs.append({
            "initial": initial_config,
            "final": final_config,
            "metric_easy": metrics["easy"],
            "metric_medium": metrics["medium"],
            "metric_hard": metrics["hard"],
        })

    return configs


"""
Function to test the models

"""
def test_models(model_g, model_p, topology, init_config_type, fixed_noise, device):
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
        data["simulated"], data["metric"] = simulate_config(data["initial"], topology, constants.n_simulation_steps, device)
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

    learning_rate_changes = [learning_rates[0]] + [prev - curr for prev, curr in zip(learning_rates[:-1], learning_rates[1:])]
    change_indices = [i for i, change in enumerate(learning_rate_changes) if change != 0]
    change_epochs = [epochs[i] for i in change_indices]
    change_lr = [learning_rates[i] for i in change_indices]

    # Plotting
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, losses_p_train, label="Training Loss", color='blue', linewidth=2, linestyle='-')
    ax1.plot(epochs, losses_p_val, label="Validation Loss", color='orange', linewidth=2, linestyle='--')
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Losses", fontsize=12)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the learning rates
    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rates, label="Learning Rate", color='green', linewidth=2, linestyle=':')
    ax2.scatter(change_epochs, change_lr, color='red', marker='o', label="LR Change")  # Mark changes
    ax2.set_ylabel("Learning Rate", fontsize=12, color='green')

    # Adjust legend to include all labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    # Save and close
    plt.savefig(os.path.join(path, "losses_graph.png"), dpi=300)  # Increase dpi for higher resolution
    plt.close()


"""
Function to check the dataset while plotting some configurations

"""
def check_train_fixed_dataset_configs():
    total_indices = 300
    n_per_png = 30
    indices = np.random.randint(0, constants.fixed_dataset_n_configs * constants.fixed_dataset_train_ratio, total_indices)
    data_name = constants.fixed_dataset_name + "_train.pt"
    dataset = torch.load(os.path.join(constants.fixed_dataset_path, data_name))

    saving_path = os.path.join(constants.fixed_dataset_path, "plots")
    os.makedirs(saving_path, exist_ok=True)

    for png_idx in range(total_indices // n_per_png):
        fig, axs = plt.subplots(n_per_png, 5, figsize=(20, 2 * n_per_png))

        for config_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + config_idx
            config = dataset[indices[global_idx]]

            for img_idx in range(5):
                ax = axs[config_idx, img_idx]
                ax.imshow(config[img_idx].detach().cpu().numpy().squeeze(), cmap='gray')
                ax.axis('off')

                if config_idx == 0:
                    titles = ["Initial", "Final", "Easy", "Medium", "Hard"]
                    ax.set_title(titles[img_idx])

        plt.tight_layout()
        plt.savefig(os.path.join(saving_path, f"fixed_configs_set_{png_idx}.png"))
        plt.close(fig)


"""
Function to check the distribution of the training dataset

"""
def check_fixed_dataset_distribution(device, dataset_path):
    n = constants.grid_size ** 2
    bins = torch.zeros(n+1, dtype=torch.int64, device=device)
    img_path = dataset_path + "_distribution.png"

    # Load the dataset
    dataset = torch.load(dataset_path, map_location=device)

    for config in dataset:
        config_flat = config[0].view(-1)
        living_cells = int(torch.sum(config_flat).item())

        if living_cells <= n:
            bins[living_cells] += 1
        else:
            print(f"Warning: Living cells count {living_cells} exceeds expected range.")

    # Move the bins tensor back to CPU for plotting
    bins_cpu = bins.to('cpu').numpy()
    max_value = bins_cpu.max()

    # Plotting
    plt.bar(range(n+1), bins_cpu)
    plt.ylim(0, max_value + max_value * 0.1)  # Add 10% headroom above the max value
    plt.draw()  # Force redraw with the new limits
    plt.savefig(img_path)
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


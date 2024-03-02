import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_conf, simulate_conf_fixed_dataset


"""
Function to get the dataloader for the training of the predictor model

Add configurations to the dataloader until it reaches the maximum number of configurations
If max number of configurations is reached, remove the oldest configurations to make room for the new ones

This method implements a sliding window approach

"""
def get_dataloader(dataloader, model_g, topology, init_conf_type, device):

    new_configs = generate_new_batches(model_g, constants.n_batches, topology, init_conf_type, device)

    if len(dataloader) + len(new_configs) > constants.n_max_batches:
        dataloader = dataloader[constants.n_batches:]

    dataloader += new_configs

    return dataloader


"""
Function to generate new batches of configurations

"""
def generate_new_batches(model_g, n_batches, topology, init_conf_type, device):

    configs = []

    for _ in range(n_batches):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)
        generated_conf = model_g(noise)
        initial_conf = get_init_conf(generated_conf, init_conf_type)

        with torch.no_grad():
            simulated_conf, simulated_metrics = simulate_conf(initial_conf, topology,
                                                             constants.n_simulation_steps, device)

        configs.append({
            "generated": generated_conf,
            "initial": initial_conf,
            "simulated":  simulated_conf,
            "metric_easy": simulated_metrics["easy"],
            "metric_medium": simulated_metrics["medium"],
            "metric_hard": simulated_metrics["hard"],
        })

    return configs


"""
Function to generate new batches of configurations for fixed dataset

"""
def generate_new_batches_fixed_dataset(model_g, n_configs, batch_size, nz, topology, init_conf_type, metric_steps, device):

    configs = []

    n_batches = n_configs // batch_size

    for _ in range(n_batches):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        generated_conf = model_g(noise)
        initial_conf = get_init_conf(generated_conf, init_conf_type)

        with torch.no_grad():
            final_conf, metrics = simulate_conf_fixed_dataset(initial_conf, topology, metric_steps, device)

        configs.append({
            "initial": initial_conf,
            "final": final_conf,
            "metric_easy": metrics["easy"],
            "metric_medium": metrics["medium"],
            "metric_hard": metrics["hard"],
        })

    return configs


"""
Function to test the models

"""
def test_models(model_g, model_p, topology, init_conf_type, fixed_noise, device):
    data = {
        "generated": None,
        "initial": None,
        "simulated": None,
        "metrics": None,
        "predicted_metric": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_conf_fixed = model_g(fixed_noise)
        data["generated"] = generated_conf_fixed
        data["initial"]   = get_init_conf(generated_conf_fixed, init_conf_type)

        data["simulated"], data["metrics"] = simulate_conf(data["initial"], topology, constants.n_simulation_steps, device)
        data["predicted_metric"] = model_p(data["initial"])

    return data


"""
Function to test the predictor model

"""
def test_predictor_model(test_set, metric_type, model_p):

    # Create an iterator from the data_loader
    data_iterator = iter(test_set)
    # Fetch the first batch
    batch = next(data_iterator)

    data = {
        "generated": None,
        "initial": None,
        "simulated": None,
        "metrics": None,
        "predicted_metric": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_p.eval()
        data["predicted_metric"] = model_p(batch[:, 0, :, :, :])

    if metric_type == constants.METRIC_TYPE["easy"]:
        metric = batch[:, 2, :, :, :]
    elif metric_type == constants.METRIC_TYPE["medium"]:
        metric = batch[:, 3, :, :, :]
    elif metric_type == constants.METRIC_TYPE["hard"]:
        metric = batch[:, 4, :, :, :]

    data["generated"] = batch[:, 0, :, :, :]
    data["initial"]   = batch[:, 0, :, :, :]
    data["simulated"] = batch[:, 1, :, :, :]
    data["metrics"]   = metric

    return data

"""
Function to save the progress plot

"""
def save_progress_plot(plot_data, epoch, results_path):

    # Convert to NumPy
    for key in plot_data.keys():
        plot_data[key] = plot_data[key].detach().cpu().numpy().squeeze()

    current_epoch = epoch+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, constants.bs-1, 4).astype(int)

    # Create figure and subplots
    fig, axs = plt.subplots(len(indices), len(plot_data), figsize=(len(indices)*len(plot_data), len(indices)*4))

    titles = ["Generated Data", "Initial Configuration", "Simulated Configuration", "Simulated Metric", "Predicted Metric"]

    plt.suptitle(f"Epoch {current_epoch}", fontsize=32)

    # Plot each data in a subplot
    for i in range(len(indices)):
        for j, key in enumerate(plot_data.keys()):
            axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray')
            axs[i, j].set_title(titles[j])

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"epoch_{current_epoch}.png"))
    plt.close(fig)


"""
Function to check the dataset while plotting some configurations

"""
def check_train_fixed_dataset_configurations():
    total_indices = 300
    n_per_png = 30
    indices = np.random.randint(0, constants.fixed_dataset_n_configs * constants.fixed_dataset_train_ratio, total_indices)
    dataset = torch.load(os.path.join(constants.fixed_dataset_path, "gol_fixed_dataset_train.pt"))

    for png_idx in range(total_indices // n_per_png):
        fig, axs = plt.subplots(n_per_png, 5, figsize=(20, 2 * n_per_png))

        for conf_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + conf_idx
            conf = dataset[indices[global_idx]]

            for img_idx in range(5):
                ax = axs[conf_idx, img_idx]
                ax.imshow(conf[img_idx].detach().cpu().numpy().squeeze(), cmap='gray')
                ax.axis('off')

                if conf_idx == 0:
                    titles = ["Initial", "Final", "Easy", "Medium", "Hard"]
                    ax.set_title(titles[img_idx])

        plt.tight_layout()
        plt.savefig(os.path.join(constants.fixed_dataset_path, f"fixed_confs_set_{png_idx}.png"))
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

    for conf in dataset:
        conf_flat = conf[0].view(-1)
        living_cells = int(torch.sum(conf_flat).item())

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
def get_init_conf(conf, init_conf_type):

    if init_conf_type == constants.INIT_CONF_TYPE["threshold"]:
        return __get_init_conf_threshold(conf)
    elif init_conf_type == constants.INIT_CONF_TYPE["n_living_cells"]:
        return __get_init_conf_n_living_cells(conf)
    else:
        raise ValueError(f"Invalid init conf type: {init_conf_type}")


"""
From the conf, get the indices of the n_living_cells highest values and set the rest to 0
The n_living_cells highest values are set to 1

"""
def __get_init_conf_n_living_cells(conf):

    init_conf = conf.clone()

    batch_size, _, height, width = init_conf.size()
    conf_flat = init_conf.view(batch_size, -1)  # Flatten each image in the batch

    # Find the indices of the top values for each image in the batch
    _, indices = torch.topk(conf_flat, constants.n_living_cells, dim=1)

    # Create a zero tensor of the same shape
    conf_new = torch.zeros_like(conf_flat)

    # Set the top indices to 1 for each image
    conf_new.scatter_(1, indices, 1)

    # Reshape back to the original shape
    init_conf = conf_new.view(batch_size, 1, height, width)

    conf = init_conf

    return conf


"""
For every value in conf, if value is < threshold, set to 0, else set to 1

"""
def __get_init_conf_threshold(conf):

    return (conf > constants.threshold_cell_value).float()


"""
Function to get a batch of a certain type of configuration from the batch itself

"""
def get_conf_from_batch(batch, type, device):
    if type == constants.CONF_NAMES["initial"]:
        return batch[:, 0, :, :, :].to(device)
    elif type == constants.CONF_NAMES["final"]:
        return batch[:, 1, :, :, :].to(device)
    elif type == constants.CONF_NAMES["metric_easy"]:
        return batch[:, 2, :, :, :].to(device)
    elif type == constants.CONF_NAMES["metric_medium"]:
        return batch[:, 3, :, :, :].to(device)
    elif type == constants.CONF_NAMES["metric_hard"]:
        return batch[:, 4, :, :, :].to(device)
    else:
        raise ValueError(f"Invalid type: {type}")

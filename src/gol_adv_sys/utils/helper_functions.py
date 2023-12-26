import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_conf


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

    for _ in range(constants.n_batches):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)
        generated_conf = model_g(noise)
        initial_conf = get_init_conf(generated_conf, init_conf_type)

        with torch.no_grad():
            simulated_conf, simulated_metric = simulate_conf(initial_conf, topology,
                                                             constants.n_simulation_steps, device)

        configs.append({
            "initial": initial_conf,
            "generated": generated_conf,
            "simulated": {"conf": simulated_conf, "metric": simulated_metric}
        })

    return configs


"""
Function to test the models

"""
def test_models(model_g, model_p, topology, init_conf_type, fixed_noise, device):
    data = {
        "generated_data": None,
        "initial_conf": None,
        "simulated_conf": None,
        "simulated_metric": None,
        "predicted_metric": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_conf_fixed   = model_g(fixed_noise)
        data["generated_data"] = generated_conf_fixed
        data["initial_conf"]   = get_init_conf(generated_conf_fixed, init_conf_type)

        data["simulated_conf"], data["simulated_metric"] = simulate_conf(data["initial_conf"],
                                                                         topology,
                                                                         constants.n_simulation_steps,
                                                                         device)
        # data["predicted_metric"] = model_p(data["generated_data"])
        data["predicted_metric"] = model_p(data["initial_conf"])

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
Function to get the elapsed time in an easily readable format

Args:
    times: list of times in seconds

All the times are summed together and then converted to hours, minutes and seconds

"""
def get_epoch_elapsed_time_str(times):

    seconds = sum(times)
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    remaining_seconds = int(math.floor(seconds % 60))

    # Format time
    time_format = f"{hours}h {minutes}m {remaining_seconds}s"

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

    return torch.where(conf < constants.threshold_cell_value, torch.zeros_like(conf), torch.ones_like(conf))


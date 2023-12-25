import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_conf


def generate_new_batches(model_g, n_batches, device):

    configs = []

    for _ in range(constants.n_batches):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)
        generated_conf = model_g(noise)

        with torch.no_grad():
            initial_conf, simulated_conf, simulated_metric = simulate_conf(generated_conf, constants.TOPOLOGY["toroidal"],
                                                                           constants.n_simulation_steps, device)

        configs.append({
            "initial": initial_conf,
            "generated": generated_conf,
            "simulated": {"conf": simulated_conf, "metric": simulated_metric}
        })

    return configs


"""
Function to get the dataloader for the training of the predictor model

Add configurations to the dataloader until it reaches the maximum number of configurations
If max number of configurations is reached, remove the oldest configurations to make room for the new ones

This method implements a sliding window approach
"""
def get_dataloader(dataloader, model_g, device):

    new_configs = generate_new_batches(model_g, constants.n_batches, device)

    if len(dataloader) + len(new_configs) > constants.n_max_batches:
        dataloader = dataloader[constants.n_batches:]

    dataloader += new_configs

    return dataloader


def test_models(model_g, model_p, fixed_noise, device):
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
        generated_conf_fixed = model_g(fixed_noise)
        data["generated_data"] = generated_conf_fixed

        data["initial_conf"] , data["simulated_conf"], data["simulated_metric"] = simulate_conf(data["generated_data"],
                                                                                                constants.TOPOLOGY["toroidal"],
                                                                                                constants.n_simulation_steps,
                                                                                                device)
        data["predicted_metric"] = model_p(data["generated_data"])

    return data


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


def get_epoch_elapsed_time_str(times):

    seconds = sum(times)
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    remaining_seconds = int(math.floor(seconds % 60))

    # Format time
    time_format = f"{hours}h {minutes}m {remaining_seconds}s"

    return time_format



import os
import matplotlib.pyplot as plt

import math
import numpy as np
import torch

from . import constants as constants
from .simulation_functions import simulate_grid


def generate_new_configs(model_g, n_configs, device):

    iters = n_configs // constants.bs
    configs = []

    for _ in range(iters):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)

        generated_grid = model_g(noise)
        simulated_grid, simulated_metric = simulate_grid(generated_grid, constants.TOPOLOGY["toroidal"], constants.n_simulation_steps, device)

        configs.append({
            "generated": generated_grid,
            "simulated": {"grid": simulated_grid, "metric": simulated_metric}
        })

    return configs


def get_dataloader(dataloader, model_g, device):

    # Add configurations to the dataloader until it reaches the maximum number of configurations
    # If max number of configurations is reached, remove the oldest configurations to make room for the new ones
    new_configs = generate_new_configs(model_g, constants.n_configs, device)
    n_to_remove = constants.n_configs // constants.bs

    # Sliding window approach
    if len(dataloader) + len(new_configs) > constants.n_max_configs // constants.bs:
        dataloader = dataloader[n_to_remove:]

    # Add the new configurations
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
        generated_grid_fixed = model_g(fixed_noise)
        data["generated_data"] = generated_grid_fixed
        data["initial_conf"]   = torch.where(generated_grid_fixed < constants.threshold_cell_value,
                                                  torch.zeros_like(generated_grid_fixed),
                                                  torch.ones_like(generated_grid_fixed))
        data["simulated_conf"], data["simulated_metric"] = simulate_grid(data["initial_conf"],
                                                                         constants.TOPOLOGY["toroidal"],
                                                                         constants.n_simulation_steps,
                                                                         device)
        data["predicted_metric"] = model_p(data["initial_conf"])

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



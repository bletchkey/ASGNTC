import os
import matplotlib.pyplot as plt

import numpy as np
import torch

from . import constants as constants


def set_grid_ncells(grid, ncells, device):

    _g = torch.zeros_like(grid)

    for j in range(constants.bs):
        flattened_grid = grid[j].flatten()
        indices_of_highest_values = torch.argsort(flattened_grid)[-ncells:]
        result_grid = torch.zeros_like(flattened_grid)
        result_grid[indices_of_highest_values] = 1.
        result_grid = result_grid.reshape_as(grid[j])
        _g[j] = result_grid

    return _g


def save_progress_plot(results_path, plot_data, epoch):

    current_epoch = epoch+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, constants.bs-1, 4).astype(int)

    # Convert to NumPy
    for key in plot_data.keys():
        plot_data[key] = plot_data[key].detach().cpu().numpy().squeeze()

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


def get_epoch_elapsed_time_minutes(times):
    __st = sum(times)
    return __st/60



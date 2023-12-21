import os
import matplotlib.pyplot as plt
import datetime

import numpy as np
import re

import torch

from . import constants as constants


def _apply_conway_rules(grid, neighbors):

    birth = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    new_grid = birth | survive

    return new_grid.float()

def _simulate_grid_toroidal(grid, kernel, device):

    # Pad the grid toroidally
    grid = torch.cat([grid[:, :, -1:], grid, grid[:, :, :1]], dim=2)
    grid = torch.cat([grid[:, :, :, -1:], grid, grid[:, :, :, :1]], dim=3)

    # Apply convolution to count neighbors
    # No additional padding is required here as we've already padded toroidally
    neighbors = torch.conv2d(grid, kernel, padding=0).to(device)

    # Remove the padding
    grid = grid[:, :, 1:-1, 1:-1]

    return _apply_conway_rules(grid, neighbors)


def _simulate_grid_flat(grid, kernel, device):

    # Apply convolution to count neighbors
    neighbors = torch.conv2d(grid, kernel, padding=1).to(device)

    return _apply_conway_rules(grid, neighbors)


def simulate_grid(grid, topology, steps, device):

    metric = torch.zeros_like(grid)
    _simulation_function = None

    # Define the simulation function
    if topology == constants.TOPOLOGY["toroidal"]:
        _simulation_function = _simulate_grid_toroidal
    elif topology == constants.TOPOLOGY["flat"]:
        _simulation_function = _simulate_grid_flat
    else:
        raise ValueError(f"Topology {topology} not supported")

    # Define the kernel for counting neighbors
    kernel = torch.ones((1, 1, 3, 3)).to(device)
    kernel[:, :, 1, 1] = 0

    for step in range(steps):
        grid = _simulation_function(grid, kernel, device)

        # Update the metric
        parameter = 0.1 * (0.999 ** step)
        metric = metric + (grid * parameter)

    return grid, metric


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


class results_folder:
    _instance = None

    @staticmethod
    def get_instance(base_path):
        if results_folder._instance is None:
            results_folder(base_path)
        return results_folder._instance.folder_path

    def __init__(self, base_path):
        if results_folder._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.folder_path = self._create_unique_date_folder(base_path)
            results_folder._instance = self

    def _create_unique_date_folder(self, base_path):
        #check base path exists
        if not os.path.exists(base_path):
            raise ValueError(f"Base path {base_path} does not exist")

        current_datetime = datetime.datetime.now()
        folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        new_folder_path = os.path.join(base_path, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        return new_folder_path


def save_progress_plot(plot_data, epoch):

    current_epoch = epoch+1
    i = 0

    # Convert plot data to numpy arrays
    plot_data["initial_conf"] = plot_data["initial_conf"][i].detach().cpu().numpy().squeeze()
    plot_data["simulated_conf"] = plot_data["simulated_conf"][i].detach().cpu().numpy().squeeze()
    plot_data["predicted_metric"] = plot_data["predicted_metric"][i].detach().cpu().numpy().squeeze()
    plot_data["simulated_metric"] = plot_data["simulated_metric"][i].detach().cpu().numpy().squeeze()

    # Create the results directory if it doesn't exist

    results_dir = results_folder.get_instance(constants.results_folder_path)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Epoch {current_epoch}")
    axs[0, 0].imshow(plot_data["initial_conf"], cmap='gray')
    axs[0, 0].set_title("Initial configuration")
    axs[0, 1].imshow(plot_data["simulated_conf"], cmap='gray')
    axs[0, 1].set_title("Simulated configuration")
    axs[1, 0].imshow(plot_data["predicted_metric"], cmap='gray')
    axs[1, 0].set_title("Predicted metric")
    axs[1, 1].imshow(plot_data["simulated_metric"], cmap='gray')
    axs[1, 1].set_title("Simulated metric")

    # Use tight layout
    plt.tight_layout()

    # Save image
    plt.savefig(os.path.join(results_dir, f"epoch_{current_epoch}.png"))

    # Close the figure
    plt.close(fig)



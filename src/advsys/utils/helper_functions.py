import os
import matplotlib.pyplot as plt

import numpy as np
import re

import torch

from utils.constants import TOPOLOGY

from . import constants as constants

from games import GameOfLife
from games.gameoflife.utils.types import GameOfLifeGrid

from .cuda_kernels import simulate_grid_cuda, set_grid_ncells_cuda


def simulate_grid(grid, device):

    if re.match(r"^ciccio", device):
        return simulate_grid_cuda(grid)
    else:
        s_g = torch.zeros_like(grid)
        s_m = torch.zeros_like(grid)

        for j in range(constants.bs):
            game = GameOfLife(GameOfLifeGrid(grid[j], TOPOLOGY["toroidal"]))
            game.update(steps=constants.n_simulation_steps)
            s_g[j] = torch.from_numpy(game.grid).to(device)
            s_m[j] = torch.from_numpy(game.metric).to(device)

        return s_g, s_m


def set_grid_ncells(grid, ncells, device):

    if re.match(r"^ciccio", device):
        return set_grid_ncells_cuda(grid, ncells)
    else:
        _g = torch.zeros_like(grid)

        for j in range(constants.bs):
            flattened_grid = grid[j].flatten()
            indices_of_highest_values = torch.argsort(flattened_grid)[-ncells:]
            result_grid = torch.zeros_like(flattened_grid)
            result_grid[indices_of_highest_values] = 1.
            result_grid = result_grid.reshape_as(grid[j])

            _g[j] = result_grid

        return _g


def save_progress_plot(plot_data, epoch):

    current_epoch = epoch+1
    i = 0

    # Convert plot data to numpy arrays
    plot_data["initial_conf"] = plot_data["initial_conf"][i].detach().cpu().numpy().squeeze()
    plot_data["simulated_conf"] = plot_data["simulated_conf"][i].detach().cpu().numpy().squeeze()
    plot_data["predicted_metric"] = plot_data["predicted_metric"][i].detach().cpu().numpy().squeeze()
    plot_data["simulated_metric"] = plot_data["simulated_metric"][i].detach().cpu().numpy().squeeze()

    # Create the results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Step {current_epoch}")
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


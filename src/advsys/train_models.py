import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import constants as constants
from .utils.helper_functions import simulate_grid, set_grid_ncells, save_progress_plot


def generate_new_configs(model_g, device):

    iters = constants.n_configs // constants.bs
    configs = []

    for _ in range(iters):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)

        # Generate a grid
        generated_grid = model_g(noise)

        # We need to set the number of living cells in the initial configuration.
        # We do this by setting the n cells with the highest values to 1 and the rest to 0
        generated_grid = set_grid_ncells(generated_grid, constants.n_max_living_cells, device)

        # Simulate the generated grid
        simulated_grid, simulated_metric = simulate_grid(generated_grid, device)

        configs.append({
            "generated": generated_grid,
            "simulated": {"grid": simulated_grid, "metric": simulated_metric}
        })

    # return the generated grid, the simulated grid and the simulated metric for the grid
    return configs


def get_dataloader(dataloader, model_g, device):

    # Add configurations to the dataloader until it reaches the maximum number of configurations
    # If max number of configurations is reached, remove the oldest configurations to make room for the new ones

    new_configs = generate_new_configs(model_g, device)
    n_to_remove = constants.n_configs // constants.bs

    # Sliding window approach
    if len(dataloader) + len(new_configs) > constants.n_max_configs // constants.bs:
        dataloader = dataloader[n_to_remove:]

    # Add the new configurations
    dataloader += new_configs

    return dataloader


def fit(model_g, model_p, optimizer_g, optimizer_p, criterion_g, criterion_p, fixed_noise, device) -> list:

    plot_data = {
        "initial_conf": None,
        "simulated_conf": None,
        "predicted_metric": None,
        "simulated_metric": None,
    }

    G_losses = []
    P_losses = []
    dataloader = []

    # Total number of epochs = num_steps * num_epochs
    for epoch in range(constants.num_epochs):

        # Get data loader for the current step of training
        dataloader = get_dataloader(dataloader, model_g, device)

        # Create a new dataloader with the configurations in a random order
        shuffled_dl = dataloader.copy()
        random.shuffle(shuffled_dl)

        print(f"\n\n---Epoch: {epoch+1}/{constants.num_epochs}---")
        print(f"Number of generated configurations in data set: {len(dataloader)*constants.bs}\n")

        # Train on the new configurations
        for step in range(constants.num_steps):

            for data in shuffled_dl:

                model_p.train()
                model_g.train()

                optimizer_g.zero_grad()
                optimizer_p.zero_grad()

                # Predict the metric for the simulated grid
                predicted_metric = model_p(data["generated"])

                # Update the generator
                errG = criterion_g(predicted_metric, data["simulated"]["metric"])
                errG.backward(retain_graph=True)
                optimizer_g.step()

                # Update the predictor
                errP = criterion_p(predicted_metric, data["simulated"]["metric"])
                errP.backward()
                optimizer_p.step()


            model_g.eval()
            model_p.eval()

            # Output training stats
            current_step = step+1

            str_step  = f"{current_step}/{constants.num_epochs}"
            str_loss_p = f"{errP.item():.4f}"
            str_loss_g = f"{errG.item():.4f}"

            print(f"Step: {str_step}, Loss P: {str_loss_p}, Loss G: {str_loss_g}")
            # Append Losses
            G_losses.append(errG.item())
            P_losses.append(errP.item())

        # Test the model on the fixed noise
        generated_grid_fixed = model_g(fixed_noise)
        plot_data["initial_conf"] = set_grid_ncells(generated_grid_fixed, constants.n_max_living_cells, device)
        plot_data["simulated_conf"], plot_data["simulated_metric"] = simulate_grid(plot_data["initial_conf"], device)
        plot_data["predicted_metric"] = model_p(plot_data["initial_conf"])

        # Save progress, once per step
        save_progress_plot(plot_data, epoch)

    return G_losses, P_losses


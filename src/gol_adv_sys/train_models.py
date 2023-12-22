import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import constants as constants
from .utils.helper_functions import simulate_grid, set_grid_ncells, save_progress_plot

def generate_new_configs(model_g, n_configs, device):

    iters = n_configs // constants.bs
    configs = []

    for _ in range(iters):
        noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device)

        # Generate a grid
        generated_grid = model_g(noise)

        # We need to set the number of living cells in the initial configuration.
        # We do this by setting the n cells with the highest values to 1 and the rest to 0
        generated_grid = set_grid_ncells(generated_grid, constants.n_max_living_cells, device)

        # Simulate the generated grid
        simulated_grid, simulated_metric = simulate_grid(generated_grid, constants.TOPOLOGY["toroidal"], constants.n_simulation_steps, device)

        configs.append({
            "generated": generated_grid,
            "simulated": {"grid": simulated_grid, "metric": simulated_metric}
        })

    # return the generated grid, the simulated grid and the simulated metric for the grid
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


def fit(model_g, model_p, optimizer_g, optimizer_p, criterion_g, criterion_p, fixed_noise, folders, device) -> list:

    torch.autograd.set_detect_anomaly(True)

    plot_data = {
        "initial_conf": None,
        "simulated_conf": None,
        "predicted_metric": None,
        "simulated_metric": None,
    }

    G_losses = []
    P_losses = []
    dataloader = []

    # Set the threshold errP, if the value is below then i can start training G
    threshold_errP_avg = 5e-06 # 0.000005

    properties_G= {"enabled": True, "can_train": False}

    # Total number of epochs = num_epochs * num_training_steps
    for epoch in range(constants.num_epochs):

        # Get data loader for the current step of training
        dataloader = get_dataloader(dataloader, model_g, device)

        # Create a new dataloader with the configurations in a random order
        shuffled_dl = dataloader.copy()
        random.shuffle(shuffled_dl)

        print(f"\n\n---Epoch: {epoch+1}/{constants.num_epochs}---")
        print(f"Number of generated configurations in data set: {len(dataloader)*constants.bs}\n")

        steps_times = [0] * constants.num_training_steps
        errs_P      = [0] * constants.num_training_steps

        # Train on the new configurations
        for step in range(constants.num_training_steps):

            step_start_time = time.time()

            # Train the generator (only if the errP is below the threshold)
            if properties_G["enabled"] and properties_G["can_train"]:
                new_configs = generate_new_configs(model_g, 4*constants.n_configs, device)

                for i in range(len(new_configs)):
                    model_g.train()
                    optimizer_g.zero_grad()
                    predicted_metric = model_p(new_configs[i]["generated"])
                    errG = criterion_g(predicted_metric, new_configs[i]["simulated"]["metric"])
                    errG.backward()
                    optimizer_g.step()


            # Train the predictor
            for i in range(len(shuffled_dl)):
                model_p.train()
                optimizer_p.zero_grad()
                predicted_metric = model_p(shuffled_dl[i]["generated"])
                errP = criterion_p(predicted_metric, shuffled_dl[i]["simulated"]["metric"])
                errs_P[step] = errP.item()
                errP.backward()
                optimizer_p.step()

            step_end_time = time.time()
            steps_times[step] = step_end_time - step_start_time

            # Set the models to eval mode
            model_p.eval()
            model_g.eval()

            # Output training stats
            current_step = step+1

            str_step   = f"{current_step}/{constants.num_training_steps}"
            str_err_p = f"{errP.item()}"

            if (properties_G["enabled"] and properties_G["can_train"]):
                str_err_g = f"{errG.item()}"
                print(f"{steps_times[step]:.2f}s | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}")
            else:
                print(f"{steps_times[step]:.2f}s | Step: {str_step}, Loss P: {str_err_p}")

            # Append Losses
            # TODO: The way G_losses is managed is temporary
            G_losses.append(errG.item() if (properties_G["enabled"] and properties_G["can_train"]) else 0)
            P_losses.append(errP.item())

        # Check if the errP is below the threshold
        errP_avg = sum(errs_P)/len(errs_P)

        if properties_G["enabled"] and not properties_G["can_train"] and errP_avg < threshold_errP_avg:
            properties_G["can_train"] = True

        # Get elapsed time for the epoch in minutes
        epoch_elapsed_time_m = sum(steps_times)/60

        # Print the total time elapsed for the epoch and the average time per step
        print(f"\nEPOCH {epoch+1} elapsed time: {epoch_elapsed_time_m:.2f} minutes")
        print(f"EPOCH {epoch+1} average time per step: {epoch_elapsed_time_m/constants.num_training_steps:.2f} minutes\n")

        # Test the model on the fixed noise after each epoch
        generated_grid_fixed = model_g(fixed_noise)
        plot_data["initial_conf"] = set_grid_ncells(generated_grid_fixed, constants.n_max_living_cells, device)
        plot_data["simulated_conf"], plot_data["simulated_metric"] = simulate_grid(plot_data["initial_conf"],
                                                                                   constants.TOPOLOGY["toroidal"],
                                                                                   constants.n_simulation_steps,
                                                                                   device)
        plot_data["predicted_metric"] = model_p(plot_data["initial_conf"])

        # Save the progress plot
        save_progress_plot(folders.results_path, plot_data, epoch)

        # Save the models
        path_g = os.path.join(folders.models_path, "generator.pth.tar")
        torch.save({
                    "state_dict": model_g.state_dict(),
                    "optimizer": optimizer_g.state_dict(),
                   }, path_g)

        path_p = os.path.join(folders.models_path, "predictor.pth.tar")
        torch.save({
                    "state_dict": model_p.state_dict(),
                    "optimizer": optimizer_p.state_dict(),
                   }, path_p)

    return G_losses, P_losses


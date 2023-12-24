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
from .utils.helper_functions import test_models, save_progress_plot, get_epoch_elapsed_time_str, generate_new_configs, get_dataloader


def fit(model_g, model_p, optimizer_g, optimizer_p, criterion_g, criterion_p, fixed_noise, folders, log_file, device) -> list:

    torch.autograd.set_detect_anomaly(True)

    G_losses = []
    P_losses = []
    dataloader = []

    path_g = os.path.join(folders.models_path, "generator.pth.tar")
    path_p = os.path.join(folders.models_path, "predictor.pth.tar")

    properties_G= {"enabled": True, "can_train": False}

    with open(log_file, "a") as log:

        # Total number of epochs = num_epochs * num_training_steps
        for epoch in range(constants.num_epochs):

            # Get data loader for the current step of training
            dataloader = get_dataloader(dataloader, model_g, device)

            # Create a new dataloader with the configurations in a random order
            shuffled_dl = dataloader.copy()
            random.shuffle(shuffled_dl)

            log.write(f"\n\nEpoch: {epoch+1}/{constants.num_epochs}")
            log.write(f"Number of generated configurations in data set: {len(dataloader)*constants.bs}\n")
            log.flush()

            steps_times = [0] * constants.num_training_steps
            errs_P      = [0] * constants.num_training_steps

            # Train on the new configurations
            for step in range(constants.num_training_steps):

                step_start_time = time.time()

                # Train the predictor
                for i in range(len(shuffled_dl)):
                    model_p.train()
                    optimizer_p.zero_grad()
                    predicted_metric = model_p(shuffled_dl[i]["generated"].detach())
                    errP = criterion_p(predicted_metric, shuffled_dl[i]["simulated"]["metric"])
                    errs_P[step] = errP.item()
                    errP.backward()
                    optimizer_p.step()

                # Train the generator (when errP_avg is below the threshold and the generator is enabled)
                if properties_G["enabled"] and properties_G["can_train"]:
                    new_configs = generate_new_configs(model_g, constants.n_configs, device)
                    for config in new_configs:
                        optimizer_g.zero_grad()
                        predicted_metric = model_p(config["generated"])
                        errG = criterion_g(predicted_metric, config["simulated"]["metric"])
                        errG.backward()
                        optimizer_g.step()


                step_end_time = time.time()
                steps_times[step] = step_end_time - step_start_time

                # Output training stats
                current_step = step+1

                str_step  = f"{current_step}/{constants.num_training_steps}"
                str_err_p = f"{errP.item()}"

                if (properties_G["enabled"] and properties_G["can_train"]):
                    str_err_g = f"{errG.item()}"
                    log.write(f"{steps_times[step]:.2f}s | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}")
                    log.flush()
                else:
                    log.write(f"{steps_times[step]:.2f}s | Step: {str_step}, Loss P: {str_err_p}")
                    log.flush()

                # Append Losses
                if properties_G["enabled"] and properties_G["can_train"]:
                    G_losses.append(errG.item())

                P_losses.append(errP.item())

            # Start training the generator if the errP_avg is below the threshold
            errP_avg = sum(errs_P)/len(errs_P)
            if properties_G["enabled"] and not properties_G["can_train"] and errP_avg < constants.threshold_avg_loss_p:
                properties_G["can_train"] = True


            # Print the total time elapsed for the epoch and the average time per step
            log.write(f"Elapsed time: {get_epoch_elapsed_time_str(steps_times)}")
            log.flush()

            # Test the models on the fixed noise
            data = test_models(model_g, model_p, fixed_noise, device)

            # Save the progress plot
            save_progress_plot(data, epoch, folders.results_path)


            # Save the models
            torch.save({
                        "state_dict": model_g.state_dict(),
                        "optimizer": optimizer_g.state_dict(),
                       }, path_g)

            torch.save({
                        "state_dict": model_p.state_dict(),
                        "optimizer": optimizer_p.state_dict(),
                       }, path_p)


            # Clear CUDA cache
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()


    return G_losses, P_losses


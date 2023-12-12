import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import constants as constants

from utils.constants import TOPOLOGY

from games import GameOfLife
from games.gameoflife.utils.types import GameOfLifeGrid


def simulate_grid(grid):

    grid = np.squeeze(grid)
    gol_grid = GameOfLifeGrid(grid, TOPOLOGY["flat"])
    game = GameOfLife(gol_grid)
    game.update(steps=constants.n_simulation_steps)

    print(game.statistics())

    return game.all_weights[-1]

def set_grid_ncells(grid):
    flattened_grid = grid.flatten()
    indices_of_highest_values = np.argsort(flattened_grid)[-constants.n_max_living_cells:]
    result_grid = np.zeros_like(flattened_grid)
    result_grid[indices_of_highest_values] = 1.
    result_grid = result_grid.reshape(grid.shape)

    return result_grid


def fit(num_epochs, model_g, model_p, optimizer_g, optimizer_p, criterion, fixed_noise, device) -> list:

    # Lists to keep track of progress
    G_losses = []
    P_losses = []
    iters = 0

    for epoch in range(num_epochs):

        for i in range(10):

            model_p.train()
            model_g.train()

            ###############################################################

            noise = torch.randn(constants.bs, constants.nz, 1, 1, device=device) # Generate batch of latent vectors
            generated_grid = model_g(noise) # Generate fake image batch with G


            #---------------------------------------------------------------

            # Create a grid of the highest values
            np_generated_grid = generated_grid.cpu().detach().numpy()
            for j in range(constants.bs):
                np_generated_grid[j] = set_grid_ncells(np_generated_grid[j])

            # Simulate the generated grid
            np_simulated_metric = np.zeros_like(np_generated_grid)
            for j in range(constants.bs):
                np_simulated_metric[j] = simulate_grid(np_generated_grid[j])


            #---------------------------------------------------------------

            """
            The generator needs to be updated based on the response of the predictor, we can implement a custom loss that reflects this objective. One way to achieve this is to create a custom loss that encourages the generator to produce an initial configuration that once it is simulated, the predictor finds it difficult to predict the simulated metric.
            """

            model_p.zero_grad()
            generated_grid = torch.from_numpy(np_generated_grid).to(device)

            guessed_metric = model_p(generated_grid).to(device)
            simulated_metric = torch.from_numpy(np_simulated_metric).to(device)

            errP = criterion(guessed_metric, simulated_metric)

            print(errP)
            errP.backward()


            model_g.eval()
            model_p.eval()


            #---------------------------------------------------------------
            # Output training stats
            # if i % 50 == 0 or i == len(train_dl)-1:
            #     current_epoch = epoch+1

            #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #           % (current_epoch, num_epochs, i, len(train_dl),
            #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # # Append Losses
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())

            iters += 1

    return G_losses, P_losses

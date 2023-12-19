import matplotlib.pyplot as plt

import numpy as np

from numba import cuda
import torch

from utils.constants import TOPOLOGY

from . import constants as constants

from games import GameOfLife
from games.gameoflife.utils.types import GameOfLifeGrid


@cuda.jit
def _simulate_grid_kernel(grid, ncells, result_grid):
    #game = GameOfLife(GameOfLifeGrid(grid, TOPOLOGY["toroidal"]))
    #game.update(steps=constants.n_simulation_steps)
    pass


def simulate_grid_cuda(grid):
   pass


@cuda.jit
def _set_grid_ncells_kernel(grid, ncells, result_grid):
    # CUDA kernel code goes here
    x, y = cuda.grid(2)  # 2D thread indices

    if x < grid.shape[0] and y < grid.shape[1]:
        # Example operation: Setting grid cell to a value based on condition
        # Modify according to your specific logic
        if grid[x, y] > ncells:
            result_grid[x, y] = 1
        else:
            result_grid[x, y] = 0


def set_grid_ncells_cuda(grid, ncells):

    # Allocate memory for 'grid' and 'result_grid' in CUDA device
    d_grid = cuda.to_device(grid)
    d_result_grid = cuda.device_array_like(grid)

    # Define blocks and threads
    threadsperblock = (16, 16)
    blockspergrid_x = (grid.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (grid.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    _set_grid_ncells_kernel[blockspergrid, threadsperblock](d_grid, ncells, d_result_grid)

    # Copy result back to host
    result_grid = d_result_grid.copy_to_host()

    return result_grid
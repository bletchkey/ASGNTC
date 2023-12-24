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

    # for every value in grid, if value is < 0.5, set to 0, else set to 1
    grid = torch.where(grid < constants.threshold_cell_value, torch.zeros_like(grid), torch.ones_like(grid))

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


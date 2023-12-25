import torch

from . import constants as constants


def _apply_conway_rules(conf, neighbors):

    birth = (neighbors == 3) & (conf == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (conf == 1)
    new_conf = birth | survive

    return new_conf.float()


def _simulate_conf_toroidal(conf, kernel, device):

    # Pad the conf toroidally
    conf = torch.cat([conf[:, :, -1:], conf, conf[:, :, :1]], dim=2)
    conf = torch.cat([conf[:, :, :, -1:], conf, conf[:, :, :, :1]], dim=3)

    # Apply convolution to count neighbors
    # No additional padding is required here as we've already padded toroidally
    neighbors = torch.conv2d(conf, kernel, padding=0).to(device)

    # Remove the padding
    conf = conf[:, :, 1:-1, 1:-1]

    return _apply_conway_rules(conf, neighbors)


def _simulate_conf_flat(conf, kernel, device):

    # Apply convolution to count neighbors
    neighbors = torch.conv2d(conf, kernel, padding=1).to(device)

    return _apply_conway_rules(conf, neighbors)


def _get_init_conf_threshold(conf):

    # for every value in conf, if value is < 0.5, set to 0, else set to 1
    return torch.where(conf < constants.threshold_cell_value, torch.zeros_like(conf), torch.ones_like(conf))


"""
From the conf, get the indices of the n_living_cells highest values and set the rest to 0
The n_living_cells highest values are set to 1

"""
def _get_init_conf_n_living_cells(conf):
    batch_size, _, height, width = conf.size()
    conf_flat = conf.view(batch_size, -1)  # Flatten each image in the batch

    # Find the indices of the top values for each image in the batch
    _, indices = torch.topk(conf_flat, constants.n_living_cells, dim=1)

    # Create a zero tensor of the same shape
    conf_new = torch.zeros_like(conf_flat)

    # Set the top indices to 1 for each image
    conf_new.scatter_(1, indices, 1)

    # Reshape back to the original shape
    conf = conf_new.view(batch_size, 1, height, width)

    return conf

    # _g = torch.zeros_like(grid)

    # for j in range(constants.bs):
    #     flattened_grid = grid[j].flatten()
    #     indices_of_highest_values = torch.argsort(flattened_grid)[-ncells:]
    #     result_grid = torch.zeros_like(flattened_grid)
    #     result_grid[indices_of_highest_values] = 1.
    #     result_grid = result_grid.reshape_as(grid[j])

    #     _g[j] = result_grid

    # return _g


def simulate_conf(conf, topology, steps, device):

    sim_metric = torch.zeros_like(conf)
    _simulation_function = None

    conf = _get_init_conf_n_living_cells(conf)
    init_config = conf.clone()

    # Define the simulation function
    if topology == constants.TOPOLOGY["toroidal"]:
        _simulation_function = _simulate_conf_toroidal
    elif topology == constants.TOPOLOGY["flat"]:
        _simulation_function = _simulate_conf_flat
    else:
        raise ValueError(f"Topology {topology} not supported")

    # Define the kernel for counting neighbors
    kernel = torch.ones((1, 1, 3, 3)).to(device)
    kernel[:, :, 1, 1] = 0

    for step in range(steps):
        conf = _simulation_function(conf, kernel, device)

        # Update the metric
        parameter = 0.1 * (0.999 ** step)
        sim_metric = sim_metric + (conf * parameter)


    return init_config, conf, sim_metric


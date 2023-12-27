import torch

from . import constants as constants


"""
Function to simulate the configuration for a given number of steps

"""
def simulate_conf(conf, topology, steps, device):

    sim_metric = torch.zeros_like(conf)
    _simulation_function = None

    # Define the simulation function
    if topology == constants.TOPOLOGY_TYPE["toroidal"]:
        _simulation_function = __simulate_conf_toroidal
    elif topology == constants.TOPOLOGY_TYPE["flat"]:
        _simulation_function = __simulate_conf_flat
    else:
        raise ValueError(f"Topology {topology} not supported")

    # Define the kernel for counting neighbors
    kernel = torch.ones((1, 1, 3, 3)).to(device)
    kernel[:, :, 1, 1] = 0

    # Limit the number of steps
    if steps > constants.n_max_simulation_steps:
        print(f"Warning: {steps} simulation steps requested, but only {constants.n_max_simulation_steps} are allowed")
        print(f"Setting steps to {constants.n_max_simulation_steps}")
        steps = constants.n_max_simulation_steps

    for step in range(steps):
        conf = _simulation_function(conf, kernel, device)

        # Update the metric
        parameter = 0.1 * (0.999 ** step)
        sim_metric = sim_metric + (conf * parameter)


    return conf, sim_metric


"""
Function to simulate the configuration for one step using the Conway's Game of Life rules
The grid is padded toroidally to simulate the toroidal topology
The padding is removed after the simulation

A convolution is applied to count the neighbors

"""
def __simulate_conf_toroidal(conf, kernel, device):

    # Pad the conf toroidally
    conf = torch.cat([conf[:, :, -1:], conf, conf[:, :, :1]], dim=2)
    conf = torch.cat([conf[:, :, :, -1:], conf, conf[:, :, :, :1]], dim=3)

    # No additional padding is required here as we've already padded toroidally
    neighbors = torch.conv2d(conf, kernel, padding=0).to(device)

    # Remove the padding
    conf = conf[:, :, 1:-1, 1:-1]

    return __apply_conway_rules(conf, neighbors)


"""
Function to simulate the configuration for one step using the Conway's Game of Life rules
Its simulates the flat topology

A convolution is applied to count the neighbors

"""
def __simulate_conf_flat(conf, kernel, device):

    neighbors = torch.conv2d(conf, kernel, padding=1).to(device)

    return __apply_conway_rules(conf, neighbors)


"""
If a cell is dead and has exactly 3 living neighbors, it becomes alive
If a cell is alive and has 2 or 3 living neighbors, it stays alive
Otherwise, it dies

"""
def __apply_conway_rules(conf, neighbors):

    birth = (neighbors == 3) & (conf == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (conf == 1)
    new_conf = birth | survive

    return new_conf.float()


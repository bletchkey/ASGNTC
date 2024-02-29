import torch

from . import constants as constants


"""
Function to simulate the configuration for a given number of steps

"""
def simulate_conf(conf, topology, steps, device):

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

    sim_confs = []
    for _ in range(steps):
        conf = _simulation_function(conf, kernel, device)
        sim_confs.append(conf)

    sim_metrics = __calculate_metrics(conf, sim_confs, steps)

    return conf, sim_metrics


"""
Function to simulate the configuration for the fixed dataset

"""
def simulate_conf_fixed_dataset(conf, topology, steps, device):

    final_conf = conf.clone()

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

    sim_confs = []
    for _ in range(steps):
        conf = _simulation_function(conf, kernel, device)
        sim_confs.append(conf)

    sim_metrics = __calculate_metrics(conf, sim_confs, steps)

    # Final conf is the last configuration before a cycle is detected
    conf_hashes = set()
    current_hash = hash(final_conf.cpu().numpy().tobytes())
    conf_hashes.add(current_hash)

    while True:
        final_conf = _simulation_function(final_conf, kernel, device)
        current_hash = hash(final_conf.cpu().numpy().tobytes())

        # Check if the current configuration has been seen before.
        if current_hash in conf_hashes:
            break

        conf_hashes.add(current_hash)


    return final_conf, sim_metrics


"""
Function for computing the metrics

"""
def __calculate_metrics(conf, confs, steps):
    sim_metrics = {
            "easy": torch.zeros_like(conf),
            "medium": torch.zeros_like(conf),
            "hard": torch.zeros_like(conf),
        }

    eps_easy   = __calculate_eps(half_step=2)
    eps_medium = __calculate_eps(half_step=4)
    eps_hard   = __calculate_eps(half_step=constants.fixed_dataset_metric_steps)

    for step, config in enumerate(reversed(confs)):
        step = len(confs) - step - 1
        sim_metrics["easy"]   = __update_metric(sim_metrics["easy"], config, step, eps_easy)
        sim_metrics["medium"] = __update_metric(sim_metrics["medium"], config, step, eps_medium)
        sim_metrics["hard"]   = __update_metric(sim_metrics["hard"], config, step, eps_hard)

    correction_easy   = eps_easy / (1 - ((1 - eps_easy) ** steps ))
    correction_medium = eps_medium / (1 - ((1 - eps_medium) ** steps))
    correction_hard   = eps_hard / (1 - ((1 - eps_hard) ** steps))

    sim_metrics["easy"]   = sim_metrics["easy"] * correction_easy
    sim_metrics["medium"] = sim_metrics["medium"] * correction_medium
    sim_metrics["hard"]   = sim_metrics["hard"] * correction_hard


    return sim_metrics


"""
Function for updating the metric

"""
def __update_metric(metric, conf, step, eps):
    return metric + (conf * ((1 - eps) ** step))


"""
Function for calculating the epsilon value

"""
def __calculate_eps(half_step):
    return 1 - (0.5 ** (1 / half_step))


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


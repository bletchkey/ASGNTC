import torch

from . import constants as constants


"""
Function to simulate the configuration for a given number of steps

"""
def simulate_config(config, topology, steps, device):

    _simulation_function = None

    # Define the simulation function
    if topology == constants.TOPOLOGY_TYPE["toroidal"]:
        _simulation_function = __simulate_config_toroidal
    elif topology == constants.TOPOLOGY_TYPE["flat"]:
        _simulation_function = __simulate_config_flat
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

    sim_configs = []
    for _ in range(steps):
        config = _simulation_function(config, kernel, device)
        sim_configs.append(config)

    sim_metrics = __calculate_metrics(config, sim_configs, steps)

    return config, sim_metrics


"""
Function to simulate the configuration for the fixed dataset

"""
def simulate_config_fixed_dataset(config, topology, steps, device):

    final_config = config.clone()

    _simulation_function = None

    # Define the simulation function
    if topology == constants.TOPOLOGY_TYPE["toroidal"]:
        _simulation_function = __simulate_config_toroidal
    elif topology == constants.TOPOLOGY_TYPE["flat"]:
        _simulation_function = __simulate_config_flat
    else:
        raise ValueError(f"Topology {topology} not supported")

    # Define the kernel for counting neighbors
    kernel = torch.ones((1, 1, 3, 3)).to(device)
    kernel[:, :, 1, 1] = 0

    sim_configs = []
    for _ in range(steps):
        config = _simulation_function(config, kernel, device)
        sim_configs.append(config)

    sim_metrics = __calculate_metrics(config, sim_configs, steps)

    # Final configuration is the last configuration before a cycle is detected
    config_hashes = set()
    current_hash = hash(final_config.cpu().numpy().tobytes())
    config_hashes.add(current_hash)

    while True:
        final_config = _simulation_function(final_config, kernel, device)
        current_hash = hash(final_config.cpu().numpy().tobytes())

        # Check if the current configuration has been seen before.
        if current_hash in config_hashes:
            break

        config_hashes.add(current_hash)


    return final_config, sim_metrics


"""
Function for computing the metrics

"""
def __calculate_metrics(config, configs, steps):
    sim_metrics = {
            "easy": torch.zeros_like(config),
            "medium": torch.zeros_like(config),
            "hard": torch.zeros_like(config),
        }

    eps_easy   = __calculate_eps(half_step=2)
    eps_medium = __calculate_eps(half_step=4)
    eps_hard   = __calculate_eps(half_step=constants.fixed_dataset_metric_steps)

    for step, config in enumerate(reversed(configs)):
        sim_metrics["easy"]   +=  (config * ((1 - eps_easy)**step))
        sim_metrics["medium"] +=  (config * ((1 - eps_medium)**step))
        sim_metrics["hard"]   +=  (config * ((1 - eps_hard)**step))

    correction_easy   = eps_easy / (1 - ((1 - eps_easy) ** steps ))
    correction_medium = eps_medium / (1 - ((1 - eps_medium) ** steps))
    correction_hard   = eps_hard / (1 - ((1 - eps_hard) ** steps))

    sim_metrics["easy"]   = sim_metrics["easy"] * correction_easy
    sim_metrics["medium"] = sim_metrics["medium"] * correction_medium
    sim_metrics["hard"]   = sim_metrics["hard"] * correction_hard


    return sim_metrics


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
def __simulate_config_toroidal(config, kernel, device):

    # Pad the config toroidally
    config = torch.cat([config[:, :, -1:], config, config[:, :, :1]], dim=2)
    config = torch.cat([config[:, :, :, -1:], config, config[:, :, :, :1]], dim=3)

    # No additional padding is required here as we've already padded toroidally
    neighbors = torch.conv2d(config, kernel, padding=0).to(device)

    # Remove the padding
    config = config[:, :, 1:-1, 1:-1]

    return __apply_conway_rules(config, neighbors)


"""
Function to simulate the configuration for one step using the Conway's Game of Life rules
Its simulates the flat topology

A convolution is applied to count the neighbors

"""
def __simulate_config_flat(config, kernel, device):

    neighbors = torch.conv2d(config, kernel, padding=1).to(device)

    return __apply_conway_rules(config, neighbors)


"""
If a cell is dead and has exactly 3 living neighbors, it becomes alive
If a cell is alive and has 2 or 3 living neighbors, it stays alive
Otherwise, it dies

"""
def __apply_conway_rules(config, neighbors):

    birth = (neighbors == 3) & (config == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (config == 1)
    new_config = birth | survive

    return new_config.float()


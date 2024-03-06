import torch
from typing import Tuple

from . import constants as constants


"""
Simulates a configuration for a given number of steps using a specified topology.

Args:
    config (torch.Tensor): The initial configuration.
    topology (str): The topology type, either "toroidal" or "flat".
    steps (int): The number of simulation steps to perform.
    calculate_final_config (bool): Whether to calculate and return the final configuration.
    device (torch.device): The computing device (CPU or GPU).

Returns:
    Tuple containing the final/simulated configuration and the dictionary of metrics for the easy, medium, and hard levels.
"""

def simulate_config(config: torch.Tensor, topology: str, steps: int,
                    calculate_final_config: bool,
                    device: torch.device) -> Tuple[torch.Tensor, dict]:

    # Mapping of topology types to their corresponding simulation functions
    simulation_functions = {
        constants.TOPOLOGY_TYPE["toroidal"]: __simulate_config_toroidal,
        constants.TOPOLOGY_TYPE["flat"]: __simulate_config_flat
    }

    # Retrieve the appropriate simulation function based on the provided topology
    _simulation_function = simulation_functions.get(topology)
    if _simulation_function is None:
        raise ValueError(f"Topology {topology} not supported")

    # Define the kernel for counting neighbors
    # Since the configuration is binary, we retrieve the number of living neighbors cells
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[:, :, 1, 1] = 0  # Set the center to 0 to exclude self

    # Optionally calculate the final configuration
    final_config = None
    if calculate_final_config == True:
        final_config = __calculate_final_configuration(config, _simulation_function, kernel, device)

    # Simulate the configuration for the given number of steps
    sim_configs = []
    for _ in range(steps):
        config = _simulation_function(config, kernel, device)
        sim_configs.append(config)

    # Calculate metrics from the simulated configurations
    metrics = __calculate_metrics(sim_configs, device)

    # Return the appropriate tuple based on whether the final configuration was calculated
    if final_config is not None:
        return (final_config, metrics)
    else:
        return (config, metrics)


"""
Function for calculating the final configuration
The final configuration is the last configuration before a cycle is detected

Args:
    config: Initial configuration
    simulation_function: Function to simulate the configuration for one step
    kernel: Kernel for computing the value of the cell using the neighbor's values
    device: Device used for computation

Returns:
    The final configuration

"""
def __calculate_final_configuration(config: torch.Tensor, simulation_function, kernel: torch.Tensor,
                                    device: torch.device) -> torch.Tensor:

    # Final configuration is the last configuration before a cycle is detected
    config_hashes = set()
    current_hash  = hash(config.cpu().numpy().tobytes())
    config_hashes.add(current_hash)

    while True:
        config = simulation_function(config, kernel, device)
        current_hash = hash(config.cpu().numpy().tobytes())

        # Check if the current configuration has been seen before.
        if current_hash in config_hashes:
            break

        config_hashes.add(current_hash)

    return config


"""
Function for calculating the metrics from the simulated configurations

Args:
    configs (list): List of simulated configurations
    device (torch.device): Device used for computation

Returns:
    Dictionary containing the metrics for the easy, medium, and hard levels.

Infos:
For numerical stability: (not used in the current implementation)

    for step in range(1, steps + 1):
        sim_metrics["easy"][-step:]   = [x * (1 - eps_easy) for x in sim_metrics["easy"][-step:]]
        sim_metrics["medium"][-step:] = [x * (1 - eps_medium) for x in sim_metrics["medium"][-step:]]
        sim_metrics["hard"][-step:]   = [x * (1 - eps_hard) for x in sim_metrics["hard"][-step:]]


"""
def __calculate_metrics(configs: list, device: torch.device) -> dict:

    steps = len(configs)

    stacked_configs = torch.stack(configs, dim=0)
    stacked_configs = stacked_configs.permute(1, 0, 2, 3, 4)

    sim_metrics = {
            "easy": stacked_configs.clone(),
            "medium": stacked_configs.clone(),
            "hard": stacked_configs.clone()
        }

    eps_easy   = __calculate_eps(half_step=2)
    eps_medium = __calculate_eps(half_step=4)
    eps_hard   = __calculate_eps(half_step=1000)

    correction_factors = {
        "easy": eps_easy / (1 - ((1 - eps_easy) ** steps)),
        "medium": eps_medium / (1 - ((1 - eps_medium) ** steps)),
        "hard": eps_hard / (1 - ((1 - eps_hard) ** steps))
    }

    # Apply decay and correction for each difficulty level
    for difficulty in ["easy", "medium", "hard"]:
        eps = locals()[f'eps_{difficulty}']
        correction = correction_factors[difficulty]

        # Efficient decay rates calculation
        step_indices = torch.arange(steps, dtype=torch.float32, device=device)
        decay_rates  = torch.pow(1 - eps, step_indices)
        decay_tensor = decay_rates.view(1, steps, 1, 1, 1)

        # Apply decay, sum, and correct
        sim_metrics[difficulty] *= decay_tensor
        sim_metrics[difficulty]  = sim_metrics[difficulty].sum(dim=1)
        sim_metrics[difficulty] *= correction

    return sim_metrics


"""
Function for calculating the epsilon value using the half step.

Args:
    half_step (int): The number of steps to reach half of the final value.

Returns:
    The epsilon value for the given half step.

"""
def __calculate_eps(half_step: int) -> float:
    return 1 - (0.5 ** (1 / half_step))


"""
Function to simulate the configuration for one step using the Conway's Game of Life rules
The grid is padded toroidally to simulate the toroidal topology
The padding is removed after the simulation

A convolution is applied to count the neighbors

Args:
    config (torch.Tensor): The current configuration.
    kernel (torch.Tensor): The kernel for counting neighbors.
    device (torch.device): The computing device (CPU or GPU).

Returns:
    The new configuration after applying the rules.

"""
def __simulate_config_toroidal(config: torch.Tensor, kernel: torch.Tensor, device: torch.device) -> torch.Tensor:

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

Args:
    config (torch.Tensor): The current configuration.
    kernel (torch.Tensor): The kernel for counting neighbors.
    device (torch.device): The computing device (CPU or GPU).

Returns:
    The new configuration after applying the rules.

"""
def __simulate_config_flat(config: torch.Tensor, kernel: torch.Tensor, device: torch.device) -> torch.Tensor:

    neighbors = torch.conv2d(config, kernel, padding=1).to(device)

    return __apply_conway_rules(config, neighbors)


"""
Function to apply the Conway's Game of Life rules:
- A cell is born if it has exactly 3 living neighbors
- A cell survives if it has 2 or 3 living neighbors
- Otherwise, the cell dies: - 0 or 1 living neighbors: underpopulation (loneliness)
                            - 4 or more living neighbors: overpopulation

Args:
    config (torch.Tensor): The current configuration.
    neighbors (torch.Tensor): The number of living neighbor's cells for each cell in the configuration.

Returns:
    The new configuration after applying the rules.

"""
def __apply_conway_rules(config: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:

    birth = (neighbors == 3) & (config == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (config == 1)
    new_config = birth | survive

    return new_config.float()


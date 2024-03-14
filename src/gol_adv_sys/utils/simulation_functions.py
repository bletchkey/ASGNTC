import torch
from typing import Tuple, Union

from . import constants as constants


def simulate_config(config: torch.Tensor, topology: str, steps: int,
                    calculate_final_config: bool,
                    device: torch.device) -> Union[Tuple[torch.Tensor, dict, int, int], Tuple[dict, dict, int, int]]:
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

        If calculate_final_config is True, the first element of the tuple is the final: a dictionary containing the final configuration, period, and anitperiod.

        If calculate_final_config is False, the first element of the tuple is the simulated configuration (torch.Tensor).

        The initial and final number of living cells are also returned.
    """
    # count living cells in the initial configuration
    n_cells_init = torch.sum(config, dim=[2, 3], dtype=torch.int32)

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
        final_config, period, antiperiod = __calculate_final_configuration(config, _simulation_function, kernel, device)

        final = {
            "config": final_config,
            "period": period,
            "antiperiod": antiperiod
        }

    # Simulate the configuration for the given number of steps
    sim_configs = []
    for _ in range(steps):
        config = _simulation_function(config, kernel, device)
        sim_configs.append(config)

    # Calculate metrics from the simulated configurations
    metrics = __calculate_metrics(sim_configs, device)

    # Return the appropriate tuple based on whether the final configuration was calculated
    if final_config is not None:
        n_cells_after = torch.sum(final_config, dim=[2, 3], dtype=torch.int32)
        return (final, metrics, n_cells_init, n_cells_after)
    else:
        n_cells_after = torch.sum(config, dim=[2, 3], dtype=torch.int32)
        return (config, metrics, n_cells_init, n_cells_after)



def __calculate_final_configuration(config_batch: torch.Tensor, simulation_function, kernel: torch.Tensor,
                                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function for calculating the final configuration
    The final configuration is the last configuration before a cycle is detected
    The period is the length of the repeating cycle
    The antiperiod is the number of steps before the cycle starts

    Args:
        config: Initial configuration
        simulation_function: Function to simulate the configuration for one step
        kernel: Kernel for computing the value of the cell using the neighbor's values
        device: Device used for computation

    Returns:
        The final configuration, period length, and antiperiod length for each configuration in the batch

    """
    batch_size = config_batch.size(0)
    final_config = torch.zeros_like(config_batch)
    period = torch.full((batch_size,), -1, dtype=torch.int32, device=device) # Initialize with -1 to indicate unfinished
    antiperiod = torch.full((batch_size,), -1, dtype=torch.int32, device=device)  # Initialize with -1 to indicate unfinished
    active = torch.ones(batch_size, dtype=torch.bool)  # Track active (unfinished) configurations

    step = 0
    config_hashes = [{} for _ in range(batch_size)]  # Separate hash tracker for each configuration

    while active.any():
        config_batch = simulation_function(config_batch, kernel, device)

        for i in range(batch_size):
            if not active[i]:
                continue  # Skip already completed configurations

            config = config_batch[i].unsqueeze(0)
            current_hash = hash(config.cpu().numpy().tobytes())

            if current_hash in config_hashes[i]:
                antiperiod[i] = config_hashes[i][current_hash]
                period[i] = step - antiperiod[i]
                final_config[i] = config.squeeze(0)
                active[i] = False
            else:
                config_hashes[i][current_hash] = step

        step += 1

    return final_config, period, antiperiod


def __calculate_metrics(configs: list, device: torch.device) -> dict:
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

    steps = len(configs)

    stacked_configs = torch.stack(configs, dim=0)
    stacked_configs = stacked_configs.permute(1, 0, 2, 3, 4)

    sim_metrics = {
        constants.METRIC_TYPE["easy"]:   stacked_configs.clone(),
        constants.METRIC_TYPE["medium"]: stacked_configs.clone(),
        constants.METRIC_TYPE["hard"]:   stacked_configs.clone()
    }

    eps = {
        constants.METRIC_TYPE["easy"]:   __calculate_eps(half_step=2),
        constants.METRIC_TYPE["medium"]: __calculate_eps(half_step=4),
        constants.METRIC_TYPE["hard"]:   __calculate_eps(half_step=1000)
    }

    correction_factor_easy   = eps[constants.METRIC_TYPE["easy"]] / (1 - ((1 - eps[constants.METRIC_TYPE["easy"]]) ** steps))
    correction_factor_medium = eps[constants.METRIC_TYPE["medium"]] / (1 - ((1 - eps[constants.METRIC_TYPE["medium"]]) ** steps))
    correction_factor_hard   = eps[constants.METRIC_TYPE["hard"]] / (1 - ((1 - eps[constants.METRIC_TYPE["hard"]]) ** steps))

    # Combine the correction factors into a dictionary
    correction_factors = {
        constants.METRIC_TYPE["easy"]:   correction_factor_easy,
        constants.METRIC_TYPE["medium"]: correction_factor_medium,
        constants.METRIC_TYPE["hard"]:   correction_factor_hard,
    }

    # Apply decay and correction for each difficulty level
    for difficulty in [constants.METRIC_TYPE["easy"], constants.METRIC_TYPE["medium"], constants.METRIC_TYPE["hard"]]:

        # Efficient decay rates calculation
        step_indices = torch.arange(steps, dtype=torch.float32, device=device)
        decay_rates  = torch.pow(1 - eps[difficulty], step_indices)
        decay_tensor = decay_rates.view(1, steps, 1, 1, 1)

        # Apply decay, sum, and correct
        sim_metrics[difficulty] *= decay_tensor
        sim_metrics[difficulty]  = sim_metrics[difficulty].sum(dim=1)
        sim_metrics[difficulty] *= correction_factors[difficulty]

    return sim_metrics


def __calculate_eps(half_step: int) -> float:
    """
    Function for calculating the epsilon value using the half step.

    Args:
        half_step (int): The number of steps to reach half of the final value.

    Returns:
        The epsilon value for the given half step.

    """
    return 1 - (0.5 ** (1 / half_step))


def __simulate_config_toroidal(config: torch.Tensor, kernel: torch.Tensor, device: torch.device) -> torch.Tensor:
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
    # Pad the config toroidally
    config = torch.cat([config[:, :, -1:], config, config[:, :, :1]], dim=2)
    config = torch.cat([config[:, :, :, -1:], config, config[:, :, :, :1]], dim=3)

    # No additional padding is required here as we've already padded toroidally
    neighbors = torch.conv2d(config, kernel, padding=0).to(device)

    # Remove the padding
    config = config[:, :, 1:-1, 1:-1]

    return __apply_conway_rules(config, neighbors)


def __simulate_config_flat(config: torch.Tensor, kernel: torch.Tensor, device: torch.device) -> torch.Tensor:
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
    neighbors = torch.conv2d(config, kernel, padding=1).to(device)

    return __apply_conway_rules(config, neighbors)


def __apply_conway_rules(config: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
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
    birth = (neighbors == 3) & (config == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (config == 1)
    new_config = birth | survive

    return new_config.float()


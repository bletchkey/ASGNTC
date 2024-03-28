import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from pathlib import Path

from configs.constants import *
from src.common.utils.simulation_functions import simulate_config


def test_models(model_g: torch.nn.Module, model_p: torch.nn.Module, topology: str,
                init_config_initial_type: str, fixed_noise: torch.Tensor, target_config: str, device: torch.device) -> dict:
    """
    Function to test the models

    Args:
        model_g (torch.nn.Module): The generator model
        model_p (torch.nn.Module): The predictor model
        topology (str): The topology to use for simulating the configurations
        init_config_initial_type (str): The type of initial configuration to use
        fixed_noise (torch.Tensor): The fixed noise to use for testing the generator model
        target_config (str): The type of configuration to predict (tipically the metric configuration)
        device (torch.device): The device used for computation

    Returns:
        dict: The dictionary containing the test results

    """

    data = {
        "generated": None,
        "initial": None,
        "simulated": None,
        "metric": None,
        "predicted": None,
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_config_fixed = model_g(fixed_noise)
        data["generated"] = generated_config_fixed
        data["initial"]   = get_initialized_initial_config(generated_config_fixed, init_config_initial_type)
        data["simulated"], sim_metrics, _, _ = simulate_config(config=data["initial"], topology=topology,
                                                         steps=N_SIM_STEPS, calculate_final_config=False,
                                                         device=device)
        data["metric"] = sim_metrics[target_config]["config"]
        data["predicted"] = model_p(data["initial"])

    return data


def save_progress_plot(plot_data: dict, epoch: int, results_path: str) -> None:
    """
    Function to save the progress plot

    Args:
        plot_data (dict): The dictionary containing the data to plot
        epoch (int): The current epoch
        results_path (str): The path to where the results will be saved

    """

    vmin = 0
    vmax = 1

    titles = []

    # Convert to NumPy
    for key in plot_data.keys():
        plot_data[key] = plot_data[key].detach().cpu().numpy().squeeze()
        titles.append(key)

    current_epoch = epoch+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, BATCH_SIZE-1, 4).astype(int)

    # Create figure and subplots
    fig, axs = plt.subplots(len(indices), len(plot_data), figsize=(len(indices)*len(plot_data), len(indices)*4))

    plt.suptitle(f"Epoch {current_epoch}", fontsize=32)

    # Plot each data in a subplot
    for i in range(len(indices)):
        for j, key in enumerate(plot_data.keys()):
            axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray', vmin=vmin, vmax=vmax)
            axs[i, j].set_title(titles[j])

    plt.tight_layout()
    plt.savefig(Path(results_path, f"epoch_{current_epoch}.png"), dpi=300)
    plt.close(fig)


def get_data_tensor(data_tensor: torch.Tensor, model_g: torch.nn.Module,
                    topology: str, init_config_initial_type: str, device: torch.device) -> torch.Tensor:
    """
    Function to get the dataloader for the training of the predictor model

    Add configurations to the dataloader until it reaches the maximum number of configurations
    If max number of configurations is reached, remove the oldest configurations to make room for the new ones

    This method implements a sliding window approach

    Args:
        data_tensor (torch.Tensor): The tensor containing the configurations
        model_g (torch.nn.Module): The generator model
        topology (str): The topology to use for simulating the configurations
        init_config_initial_type (str): The type of initial configuration to use
        device (torch.device): The device to use for computation

    Returns:
        torch.Tensor: The updated data tensor

    """
    new_configs = generate_new_batches(model_g, N_BATCHES, topology, init_config_initial_type, device)

    # If data_tensor is None, initialize it with new_configs
    if data_tensor is None:
        data_tensor = new_configs
    else:
        # Concatenate current data with new_configs
        combined_tensor = torch.cat([data_tensor, new_configs], dim=0)

        # If the combined size exceeds the max allowed size, trim the oldest entries
        max_size = N_MAX_BATCHES * BATCH_SIZE
        if combined_tensor.size(0) > max_size:
            # Calculate number of entries to drop from the start to fit the new_configs
            excess_entries = combined_tensor.size(0) - max_size
            data_tensor = combined_tensor[excess_entries:]
        else:
            data_tensor = combined_tensor

    logging.info(f"Data tensor shape: {data_tensor.shape}, used for creating the dataloader.")

    return data_tensor


def generate_new_batches(model_g: torch.nn.Module, n_batches: int, topology: str,
                         init_config_initial_type: str, device: torch.device) -> torch.Tensor:

    """
    Function to generate new batches of configurations

    Args:
        model_g (torch.nn.Module): The generator model
        n_batches (int): The number of batches to generate
        topology (str): The topology to use for simulating the configurations
        init_config_initial_type (str): The type of initial configuration to use
        device (torch.device): The device used for computation

    Returns:
        torch.Tensor: The tensor containing the new configurations

    """

    configs = []

    for _ in range(n_batches):
        noise = torch.randn(BATCH_SIZE, N_Z, 1, 1, device=device)
        generated_config = model_g(noise)
        initial_config = get_initialized_initial_config(generated_config, init_config_initial_type)

        with torch.no_grad():
            simulated_config, metrics, _, _ = simulate_config(config=initial_config, topology=topology,
                                                              steps=N_SIM_STEPS, calculate_final_config=False,
                                                              device=device)
        configs.append({
            CONFIG_INITIAL: initial_config,
            CONFIG_SIMULATED:  simulated_config,
            CONFIG_METRIC_EASY: metrics[CONFIG_METRIC_EASY]["config"],
            CONFIG_METRIC_MEDIUM: metrics[CONFIG_METRIC_MEDIUM]["config"],
            CONFIG_METRIC_HARD: metrics[CONFIG_METRIC_HARD]["config"],
            CONFIG_METRIC_STABLE: metrics[CONFIG_METRIC_STABLE]["config"]
        })

    # Create a tensor from the list of configurations
    concatenated_tensors = []
    keys = configs[0].keys()
    for key in keys:
        concatenated_tensor = torch.cat([config[key] for config in configs], dim=0)
        concatenated_tensors.append(concatenated_tensor)
    # Stack the tensors
    data = torch.stack(concatenated_tensors, dim=1)
    # Shuffle the data
    data = data[torch.randperm(data.size(0))]

    return data


def get_initialized_initial_config(config: torch.Tensor, init_config_initial_type: str) -> torch.Tensor:
    """
    Function to get the initial configuration from the generated configuration

    Args:
        config (torch.Tensor): The generated configuration
        init_config_initial_type (str): The type of initial configuration to use

    Returns:
        torch.Tensor: The initial configuration

    """
    if init_config_initial_type == INIT_CONFIG_INTIAL_THRESHOLD:
        return __initialize_config_threshold(config)
    elif init_config_initial_type == INIT_CONFIG_INITAL_N_CELLS:
        return __initialize_config_n_living_cells(config)
    else:
        raise ValueError(f"Invalid init configuration type: {init_config_initial_type}")


def __initialize_config_n_living_cells(config: torch.Tensor) -> torch.Tensor:
    """
    From the configuration, get the indices of the n_living_cells highest values and set the rest to 0
    The n_living_cells highest values are set to 1

    Args:
        config (torch.Tensor): The configuration

    Returns:
        updated_config (torch.Tensor): The updated configuration

    """
    batch_size, channels, height, width = config.size()
    config_flat = config.view(batch_size, -1)  # Flatten each image in the batch

    # Find the indices of the top values for each image in the batch
    _, indices = torch.topk(config_flat, N_LIVING_CELLS_VALUE, dim=1)

    # Create a zero tensor of the same shape
    updated_config = torch.zeros_like(config_flat)

    # Set the top indices to 1 for each image
    updated_config.scatter_(1, indices, 1)

    # Reshape back to the original shape
    updated_config = updated_config.view(batch_size, channels, height, width)

    return updated_config


def __initialize_config_threshold(config: torch.Tensor) -> torch.Tensor:
    """
    For every value in configuration, if value is < threshold, set to 0, else set to 1

    Args:
        config (torch.Tensor): The configuration

    Returns:
        torch.Tensor: The updated configuration

    """
    return (config > THRESHOLD_CELL_VALUE).float()


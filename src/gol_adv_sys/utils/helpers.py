import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from typing import Tuple
from pathlib import Path

from configs.constants        import *
from src.common.utils.helpers import export_figures_to_pdf

from src.common.utils.simulation_functions import simulate_config
from src.common.utils.scores               import calculate_stable_target_complexity


def test_models_DCGAN(model_g: torch.nn.Module,
                      model_p: torch.nn.Module,
                      topology: str,
                      fixed_noise: torch.Tensor,
                      target_config: str,
                      device: torch.device) -> dict:
    """
    Function to test the models

    Args:
        model_g (torch.nn.Module): The generator model
        model_p (torch.nn.Module): The predictor model
        topology (str): The topology to use for simulating the configurations
        fixed_noise (torch.Tensor): The fixed noise to use for testing the generator model
        target_config (str): The type of configuration to predict (tipically the target configuration)
        device (torch.device): The device used for computation

    Returns:
        dict: The dictionary containing the test results

    """

    data = {
        "generated": None,
        "initial": None,
        "final": None,
        "simulated": None,
        "target": None,
        "predicted": None,
        "metadata" : None
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_config_fixed = model_g(fixed_noise)
        data["generated"] = generated_config_fixed
        data["initial"]   = 0.5 + 0.5*torch.sign(generated_config_fixed)
        sim_results = simulate_config(config=data["initial"], topology=topology,
                                      steps=N_SIM_STEPS, device=device)

        data["final"]     = sim_results["final"]
        data["simulated"] = sim_results["simulated"]
        data["target"]    = sim_results["all_targets"][target_config]["config"]
        data["predicted"] = model_p(data["initial"])
        data["metadata"]  = {
            "n_cells_initial":   sim_results["n_cells_initial"],
            "n_cells_simulated": sim_results["n_cells_simulated"],
            "n_cells_final":     sim_results["n_cells_final"],
            "transient_phase":   sim_results["transient_phase"],
            "period":            sim_results["period"],
            "target_minmum":     sim_results["all_targets"][target_config]["minimum"],
            "target_maximum":    sim_results["all_targets"][target_config]["maximum"],
            "target_q1":         sim_results["all_targets"][target_config]["q1"],
            "target_q2":         sim_results["all_targets"][target_config]["q2"],
            "target_q3":         sim_results["all_targets"][target_config]["q3"],
            "target_name":       target_config
        }

    return data


def test_models(model_g: torch.nn.Module, model_p: torch.nn.Module,
                topology: str, init_config_initial_type: str,
                fixed_input: torch.Tensor, target_config: str, device: torch.device) -> dict:
    """
    Function to test the models

    Args:
        model_g (torch.nn.Module): The generator model
        model_p (torch.nn.Module): The predictor model
        topology (str): The topology to use for simulating the configurations
        init_config_initial_type (str): The type of initial configuration to use
        fixed_input (torch.Tensor): The fixed input to use for testing the generator model
        target_config (str): The type of configuration to predict (tipically the target configuration)
        device (torch.device): The device used for computation

    Returns:
        dict: The dictionary containing the test results

    """

    data = {
        "generated": None,
        "initial": None,
        "final": None,
        "simulated": None,
        "target": None,
        "predicted": None,
        "metadata" : None
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()

        generated_config_fixed, _ = model_g(fixed_input)
        data["generated"] = generated_config_fixed
        data["initial"]   = generated_config_fixed
        sim_results = simulate_config(config=data["initial"], topology=topology,
                                      steps=N_SIM_STEPS, device=device)

        data["final"]     = sim_results["final"]
        data["simulated"] = sim_results["simulated"]
        data["target"]    = sim_results["all_targets"][target_config]["config"]
        data["predicted"] = model_p(data["initial"])
        data["metadata"]  = {
            "n_cells_initial":   sim_results["n_cells_initial"],
            "n_cells_simulated": sim_results["n_cells_simulated"],
            "n_cells_final":     sim_results["n_cells_final"],
            "transient_phase":   sim_results["transient_phase"],
            "period":            sim_results["period"],
            "target_minmum":     sim_results["all_targets"][target_config]["minimum"],
            "target_maximum":    sim_results["all_targets"][target_config]["maximum"],
            "target_q1":         sim_results["all_targets"][target_config]["q1"],
            "target_q2":         sim_results["all_targets"][target_config]["q2"],
            "target_q3":         sim_results["all_targets"][target_config]["q3"],
            "target_name":       target_config
        }

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
        if isinstance(plot_data[key], torch.Tensor):
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

            if key != "metadata":
                axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray', vmin=vmin, vmax=vmax)

            if key == "metadata":
                text_data = ""
                for k in plot_data[key].keys():
                    if k.startswith("n_cells"):
                        continue
                    elif k.startswith("target") and k != "target_name":
                        value = plot_data[key][k][indices[i]]
                        text_data += f"{k}: {value:.4f}\n"
                    elif k == "target_name":
                        continue
                    else:
                        text_data += f"{k}: {plot_data[key][k][indices[i]]}\n"
                axs[i, j].text(0.1, 0.5, text_data, fontsize=18, ha='left', va='center', transform=axs[i, j].transAxes)
                if i == 0:
                    axs[i, j].set_title(titles[j])

            elif key == "initial":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['n_cells_initial'][indices[i]].item()} cells")
            elif key == "simulated":
                axs[i, j].set_title(titles[j] + f" - {N_SIM_STEPS} steps - {plot_data['metadata']['n_cells_simulated'][indices[i]].item()} cells")
            elif key == "final":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['n_cells_final'][indices[i]].item()} cells")
            elif key == "target":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['target_name']} ")
            else:
                axs[i, j].set_title(titles[j])

            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    pdf_path = Path(results_path, f"epoch_{current_epoch}.pdf")
    export_figures_to_pdf(pdf_path, fig)


def get_config_from_batch(batch: torch.Tensor, type: str, device: torch.device) -> torch.Tensor:
    """
    Function to get a batch of a certain type of configuration from the batch itself

    Args:
        batch (torch.Tensor): The batch containing the configurations
        type (str): The type of configuration to retrieve
        device (torch.device): The device to use for computation

    Returns:
        torch.Tensor: The configuration specified by the type

    """
    # Ensure the batch has the expected dimensions (5D tensor)
    if batch.dim() != 5:
        raise RuntimeError(f"Expected batch to have 5 dimensions, got {batch.dim()}")

    # Mapping from type to index in the batch
    config_indices = {
        CONFIG_INITIAL       : 0,
        CONFIG_GENERATED     : 1,
        CONFIG_SIMULATED     : 2,
        CONFIG_FINAL         : 3,
        CONFIG_TARGET_EASY   : 4,
        CONFIG_TARGET_MEDIUM : 5,
        CONFIG_TARGET_HARD   : 6,
        CONFIG_TARGET_STABLE : 7
    }

    # Validate and retrieve the configuration index
    if type not in config_indices:
        raise ValueError(f"Invalid type: {type}. Valid types are {list(config_indices.keys())}")

    config_index = config_indices[type]

    # Extract and return the configuration
    return batch[:, config_index, :, :, :].to(device)


def get_data_tensor(data_tensor: torch.Tensor,
                    model_g: torch.nn.Module,
                    topology: str,
                    device: torch.device) -> torch.Tensor:
    """
    Function to get the dataloader for the training of the predictor model

    Add configurations to the dataloader until it reaches the maximum number of configurations
    If max number of configurations is reached, remove the oldest configurations to make room for the new ones

    This method implements a sliding window approach

    Args:
        data_tensor (torch.Tensor): The tensor containing the configurations
        model_g (torch.nn.Module): The generator model
        topology (str): The topology to use for simulating the configurations
        device (torch.device): The device to use for computation

    Returns:
        torch.Tensor: The updated data tensor

    """

    new_configs = generate_new_batches(model_g, N_BATCHES, topology, device)

    # Calculate the average complexity of the stable targets
    stable_targets               = get_config_from_batch(new_configs, CONFIG_TARGET_STABLE, device)
    avg_stable_target_complexity = calculate_stable_target_complexity(stable_targets, mean=True)

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

    return data_tensor, avg_stable_target_complexity


def generate_new_batches(model_g: torch.nn.Module,
                         n_batches: int,
                         topology: str,
                         device: torch.device) -> torch.Tensor:

    """
    Function to generate new batches of configurations

    Args:
        model_g (torch.nn.Module): The generator model
        n_batches (int): The number of batches to generate
        topology (str): The topology to use for simulating the configurations
        device (torch.device): The device used for computation

    Returns:
        torch.Tensor: The tensor containing the new configurations

    """

    configs = []

    for _ in range(n_batches):

        generated_config = __generate_initial_config_from_noise(model_g, device)

        with torch.no_grad():
            # Make sure the initial configuration is only 0 or 1
            initial_config = 0.5 + 0.5 * torch.sign(generated_config)

            sim_results = simulate_config(config=initial_config, topology=topology,
                                          steps=N_SIM_STEPS, device=device)

        simulated_config = sim_results["simulated"]
        targets          = sim_results["all_targets"]
        final_config     = sim_results["final"]

        configs.append({
            CONFIG_INITIAL       : initial_config,
            CONFIG_GENERATED     : generated_config,
            CONFIG_SIMULATED     : simulated_config,
            CONFIG_FINAL         : final_config,
            CONFIG_TARGET_EASY   : targets[CONFIG_TARGET_EASY]["config"],
            CONFIG_TARGET_MEDIUM : targets[CONFIG_TARGET_MEDIUM]["config"],
            CONFIG_TARGET_HARD   : targets[CONFIG_TARGET_HARD]["config"],
            CONFIG_TARGET_STABLE : targets[CONFIG_TARGET_STABLE]["config"]
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


def __generate_initial_config_from_noise(model_g: torch.nn.Module,
                                         device: torch.device) -> torch.Tensor:

    """
    Function to generate the initial configuration

    Args:
        model_g (torch.nn.Module): The generator model
        device (torch.device)    : The device used for computation

    Returns:
        torch.Tensor: The initial configuration

    """
    noise            = torch.randn(BATCH_SIZE, N_Z, 1, 1,
                                   device=device)
    generated_config = model_g(noise)


    return generated_config


def __generate_initial_config(model_g: torch.nn.Module,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Function to generate the initial configuration

    Args:
        model_g (torch.nn.Module): The generator model
        device (torch.device): The device used for computation

    Returns:
        torch.Tensor: The initial configuration

    Infos:

    Previuosly, the initial configuration was generated using a noise tensor.
    This procedure was applied when generating the initial configuration with the generator model
    that used transposed convolutions.

        noise = torch.randn(BATCH_SIZE, N_Z, 1, 1, device=device)
        generated_config = model_g(noise)
        initial_config = get_initialized_initial_config(generated_config, init_config_initial_type)

    """

    input_config = torch.zeros(BATCH_SIZE, GRID_NUM_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
    input_config[:, :, GRID_SIZE // 2, GRID_SIZE // 2] = 1

    generated_config, probabilities = model_g(input_config)

    return generated_config, probabilities


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
    _, indices = torch.topk(config_flat, N_LIVING_CELLS_INITIAL, dim=1)

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


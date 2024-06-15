import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist
from typing import Tuple
from pathlib import Path

from configs.constants        import *
from src.common.utils.helpers import export_figures_to_pdf

from src.common.utils.simulation_functions import simulate_config, adv_training_simulate_config
from src.common.utils.scores               import calculate_stable_target_complexity


def test_models(model_g: torch.nn.Module,
                model_p: torch.nn.Module,
                topology: str,
                init_config_initial_type: str,
                fixed_input_noise: torch.Tensor,
                target_config: str,
                num_sim_steps: int,
                predictor_device: torch.device,
                generator_device: torch.device) -> dict:
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
        "noise"    : None,
        "generated": None,
        "initial"  : None,
        "final"    : None,
        "simulated": None,
        "target"   : None,
        "predicted": None,
        "metadata" : None
    }

    # Test the models on the fixed noise
    with torch.no_grad():
        model_g.eval()
        model_p.eval()
        generated_config_fixed = model_g(fixed_input_noise)
        data["generated"]      = generated_config_fixed
        data["initial"]        = get_initial_config(generated_config_fixed, init_config_initial_type)

        sim_results = simulate_config(config=data["initial"],
                                      topology=topology,
                                      steps=num_sim_steps,
                                      device=generator_device)

        data["noise"]     = fixed_input_noise
        data["final"]     = sim_results["final"]
        data["simulated"] = sim_results["simulated"]
        data["target"]    = sim_results["all_targets"][target_config]["config"]
        data["predicted"] = model_p(data["generated"].to(predictor_device))
        data["metadata"]  = {
            "n_cells_initial"  : sim_results["n_cells_initial"],
            "n_cells_simulated": sim_results["n_cells_simulated"],
            "n_cells_final"    : sim_results["n_cells_final"],
            "transient_phase"  : sim_results["transient_phase"],
            "period"           : sim_results["period"],
            "target_min"       : sim_results["all_targets"][target_config]["minimum"],
            "target_max"       : sim_results["all_targets"][target_config]["maximum"],
            "target_q1"        : sim_results["all_targets"][target_config]["q1"],
            "target_q2"        : sim_results["all_targets"][target_config]["q2"],
            "target_q3"        : sim_results["all_targets"][target_config]["q3"],
            "target_name"      : target_config
        }

    return data


def save_progress_plot(plot_data: dict,
                       iteration: int,
                       num_sim_steps: int,
                       results_path: str) -> None:
    """
    Function to save the progress plot

    Args:
        plot_data (dict)  : The dictionary containing the data to plot
        iteration (int)   : The current iteration
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


    current_iteration = iteration+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, ADV_BATCH_SIZE-1, 4).astype(int)

    # Create figure and subplots
    fig, axs = plt.subplots(len(indices), len(plot_data), figsize=(len(indices)*len(plot_data), len(indices)*4))

    # Plot each data in a subplot
    for i in range(len(indices)):
        for j, key in enumerate(plot_data.keys()):

            if key != "metadata":
                axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray', vmin=vmin, vmax=vmax)

            if key == "metadata":
                axs[i, j].set_title(titles[j], fontsize=12, fontweight='bold')
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
                axs[i, j].axis('off')

            elif key == "noise":
                axs[i, j].set_title(titles[j] + f" - Dirichlet alpha: {DIRICHLET_ALPHA}", fontsize=12, fontweight='bold')
            elif key == "initial":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['n_cells_initial'][indices[i]].item()} cells", fontsize=12, fontweight='bold')
            elif key == "simulated":
                axs[i, j].set_title(titles[j] + f" - {num_sim_steps} steps - {plot_data['metadata']['n_cells_simulated'][indices[i]].item()} cells", fontsize=12, fontweight='bold')
            elif key == "final":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['n_cells_final'][indices[i]].item()} cells", fontsize=12, fontweight='bold')
            elif key == "target":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['target_name']}", fontsize=12, fontweight='bold')
            else:
                axs[i, j].set_title(titles[j], fontsize=12, fontweight='bold')

            # border colors
            for spine in axs[i, j].spines.values():
                spine.set_edgecolor('#000000')
                spine.set_linewidth(2)

            # Remove ticks from x and y axes
            axs[i, j].tick_params(axis='both',          # Changes apply to both x and y axes
                                  which='both',         # Affects both major and minor ticks
                                  left=False,           # Remove left ticks
                                  right=False,          # Remove right ticks
                                  bottom=False,         # Remove bottom ticks
                                  top=False,            # Remove top ticks
                                  labelleft=False,      # Remove labels on the left
                                  labelbottom=False)    # Remove labels on the bottom

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    pdf_path = Path(results_path, f"iteration_{current_iteration}.pdf")
    export_figures_to_pdf(pdf_path, fig)


def save_progress_graph(stats: dict,
                        results_path: str) -> None:
    """
    Function to save the progress graph.

    Args:
        stats (dict)      : The dictionary containing the stats data to plot.
        results_path (str): The path to where the results will be saved.

    """

    # Extract stats
    n_cells_initial  = stats["n_cells_initial"]
    n_cells_final    = stats["n_cells_final"]
    prediction_score = stats["prediction_score"]
    period           = stats["period"]
    transient_phase  = stats["transient_phase"]
    iterations       = stats["iterations"]

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot the number of cells initial and final
    axs[0, 0].plot(iterations, n_cells_initial, label="Initial config", color="#366926", linestyle=":", marker="x", linewidth=1.5)
    axs[0, 0].plot(iterations, n_cells_final, label="Final config", color="#821631", linestyle=":", marker="x", linewidth=1.5)
    axs[0, 0].set_title("Number of cells", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("iterations", fontsize=10)
    axs[0, 0].set_ylabel("number of cells", fontsize=10)
    axs[0, 0].legend(loc='upper left')
    axs[0, 0].grid(True)

    # Plot the prediction score
    axs[0, 1].plot(iterations, prediction_score, label="Prediction score", color="#385BA8", linestyle=":", marker="x", linewidth=1.5)
    axs[0, 1].set_title("Prediction scores", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("iterations", fontsize=10)
    axs[0, 1].set_ylabel("score (%)", fontsize=10)
    axs[0, 1].set_ylim([0, 100])
    axs[0, 1].margins(y=0.05)
    axs[0, 1].legend(loc='upper left')
    axs[0, 1].grid(True)

    # Plot the period
    axs[1, 0].plot(iterations, period, label="Period", color="#512275", linestyle=":", marker="x", linewidth=1.5)
    axs[1, 0].set_title("Periods", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("iterations", fontsize=10)
    axs[1, 0].set_ylabel("period length", fontsize=10)
    axs[1, 0].legend(loc='upper left')
    axs[1, 0].grid(True)

    # Plot the transient phase
    axs[1, 1].plot(iterations, transient_phase, label="Transient phase", color="#ad6d2d", linestyle=":", marker="x", linewidth=1.5)
    axs[1, 1].set_title("Transient phases", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("iterations", fontsize=10)
    axs[1, 1].set_ylabel("transient phase length", fontsize=10)
    axs[1, 1].legend(loc='upper left')
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()

    pdf_path = Path(results_path, f"iterations_progress_graph.pdf")
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


def get_config_from_training_batch(batch: torch.Tensor, type: str, device: torch.device) -> torch.Tensor:
    """
    Function to get a batch of a certain type of configuration from the training batch

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
        CONFIG_GENERATED     : 0,
        CONFIG_TARGET_EASY   : 1,
        CONFIG_TARGET_MEDIUM : 1,
        CONFIG_TARGET_HARD   : 1,
        CONFIG_TARGET_STABLE : 1
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
                    num_sim_steps: int,
                    init_config_initial_type: str,
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

    new_configs = generate_new_batches(model_g,
                                       NUM_BATCHES,
                                       topology,
                                       init_config_initial_type,
                                       num_sim_steps,
                                       device)

    # Calculate the average complexity of the stable targets

    # stable_targets               = get_config_from_batch(new_configs, CONFIG_TARGET_STABLE, device)
    # avg_stable_target_complexity = calculate_stable_target_complexity(stable_targets, mean=True)

    # If data_tensor is None, initialize it with new_configs
    if data_tensor is None:
        data_tensor = new_configs
    else:
        future_combination_size = data_tensor.size(0) + new_configs.size(0)

        # If the combined size exceeds the max allowed size, trim the oldest entries
        if future_combination_size > NUM_MAX_BATCHES * ADV_BATCH_SIZE:
            # Calculate number of entries to drop from the start to fit the new_configs
            n_entries_to_drop = future_combination_size - NUM_MAX_BATCHES * ADV_BATCH_SIZE
            data_tensor = data_tensor[n_entries_to_drop:, :, :, :, :]
            data_tensor = torch.cat([data_tensor, new_configs], dim=0)
        else:
            data_tensor = torch.cat([data_tensor, new_configs], dim=0)

    data_tensor = data_tensor

    return data_tensor


def generate_new_batches(model_g: torch.nn.Module,
                         n_batches: int,
                         topology: str,
                         num_sim_steps: int,
                         init_config_initial_type: str,
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

        generated_config = get_generated_config(model_g, device)

        with torch.no_grad():
            initial_config = get_initial_config(generated_config, init_config_initial_type)
            sim_results    = simulate_config(config=initial_config, topology=topology,
                                             steps=num_sim_steps, device=device)


        targets = sim_results["all_targets"]

        configs.append({
            CONFIG_INITIAL       : initial_config,
            CONFIG_GENERATED     : generated_config,
            CONFIG_SIMULATED     : sim_results["simulated"],
            CONFIG_FINAL         : sim_results["final"],
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


def generate_new_training_batches(model_g: torch.nn.Module,
                                  n_batches: int,
                                  topology: str,
                                  target_type: str,
                                  num_sim_steps: int,
                                  init_config_initial_type: str,
                                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:

    # Preallocate tensors based on the total size needed
    total_data_count = n_batches * ADV_BATCH_SIZE
    data_size        = (NUM_CHANNELS_GRID, GRID_SIZE, GRID_SIZE)
    generated_tensor = torch.zeros((total_data_count, *data_size), device=device)
    target_tensor    = torch.zeros((total_data_count, *data_size), device=device)

    index = 0
    for _ in range(n_batches):
        generated_config = get_generated_config(model_g, device)

        with torch.no_grad():
            initial_config = get_initial_config(generated_config, init_config_initial_type)
            target = adv_training_simulate_config(config=initial_config,
                                                          topology=topology,
                                                          steps=num_sim_steps,
                                                          target_type=target_type,
                                                          device=device)

        # Insert the generated data into the preallocated tensor
        generated_tensor[index:index + ADV_BATCH_SIZE] = generated_config
        target_tensor[index:index + ADV_BATCH_SIZE]    = target
        index += ADV_BATCH_SIZE

    return generated_tensor, target_tensor


def get_generated_config(model_g: torch.nn.Module,
                         device: torch.device,
                         noise_vector: bool = False) -> torch.Tensor:

    """
    Function to generate the initial configuration

    Args:
        model_g (torch.nn.Module): The generator model
        device (torch.device)    : The device used for computation

    Returns:
        torch.Tensor: The generated initial configuration

    """

    if noise_vector == True:
        noise            = torch.randn(ADV_BATCH_SIZE, LATENT_VEC_SIZE, 1, 1, device=device)

        generated_config = model_g(noise)

        return generated_config

    input_config     = get_dirichlet_input_noise(ADV_BATCH_SIZE, device)
    generated_config = model_g(input_config)

    return generated_config


def get_dirichlet_input_noise(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Function to generate the input noise for the generator model

    Args:
        batch_size (int): The batch size
        device (torch.device): The device used for computation

    Returns:
        torch.Tensor: The input noise for the generator model

    """
    concentration = torch.ones([GRID_SIZE, GRID_SIZE], device=device) * DIRICHLET_ALPHA
    input_config  = dist.Dirichlet(concentration).sample((batch_size, NUM_CHANNELS_GRID))

    return input_config


def get_initial_config(config: torch.Tensor,
                       init_config_initial_type: str) -> torch.Tensor:
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
    elif init_config_initial_type == INIT_CONFIG_INITIAL_SIGN:
        return __initialize_config_sign(config)
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
    _, indices = torch.topk(config_flat, NUM_LIVING_CELLS_INITIAL, dim=1)

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


def __initialize_config_sign(config: torch.Tensor) -> torch.Tensor:
    """
    For every value in configuration, if value is < 0, set to 0, else set to 1

    Args:
        config (torch.Tensor): The configuration

    Returns:
        torch.Tensor: The updated configuration

    """
    return 0.5 + 0.5 * torch.sign(config)


# -------------- DCGAN ----------------


def test_models_DCGAN(model_g: torch.nn.Module,
                      model_p: torch.nn.Module,
                      topology: str,
                      fixed_noise: torch.Tensor,
                      target_config: str,
                      num_sim_steps: int,
                      init_config_initial_type: str,
                      device: torch.device) -> dict:
    """
    Function to test the models

    Args:
        model_g (torch.nn.Module): The generator model
        model_p (torch.nn.Module): The predictor model
        topology (str): The topology to use for simulating the configurations
        fixed_noise (torch.Tensor): The fixed noise to use for testing the generator model
        target_config (str): The type of configuration to predict (tipically the target configuration)
        init_config_initial_type (str): The type of initilization for the initial configuration to use.
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
        data["generated"]      = generated_config_fixed
        data["initial"]   = get_initial_config(generated_config_fixed, init_config_initial_type)
        sim_results       = simulate_config(config=data["initial"], topology=topology,
                                            steps=num_sim_steps, device=device)

        data["final"]     = sim_results["final"]
        data["simulated"] = sim_results["simulated"]
        data["target"]    = sim_results["all_targets"][target_config]["config"]
        data["predicted"] = model_p(data["generated"])
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


def save_progress_plot_DCGAN(plot_data: dict,
                             iteration: int,
                             num_sim_steps: int,
                             results_path: str) -> None:
    """
    Function to save the progress plot

    Args:
        plot_data (dict)  : The dictionary containing the data to plot
        iteration (int)   : The current iteration
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

    current_iteration = iteration+1

    # Get 4 equally spaced indices
    indices = np.linspace(0, ADV_BATCH_SIZE-1, 4).astype(int)

    # Create figure and subplots
    fig, axs = plt.subplots(len(indices), len(plot_data), figsize=(len(indices)*len(plot_data), len(indices)*4))

    plt.suptitle(f"Iteration {current_iteration}", fontsize=32)

    # Plot each data in a subplot
    for i in range(len(indices)):
        for j, key in enumerate(plot_data.keys()):

            if key != "metadata":
                axs[i, j].imshow(plot_data[key][indices[i]], cmap='gray', vmin=vmin, vmax=vmax)

                for spine in axs[i, j].spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

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
                axs[i, j].set_title(titles[j] + f" - {num_sim_steps} steps - {plot_data['metadata']['n_cells_simulated'][indices[i]].item()} cells")
            elif key == "final":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['n_cells_final'][indices[i]].item()} cells")
            elif key == "target":
                axs[i, j].set_title(titles[j] + f" - {plot_data['metadata']['target_name']} ")
            else:
                axs[i, j].set_title(titles[j])

            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    pdf_path = Path(results_path, f"inter_{current_iteration}.pdf")
    export_figures_to_pdf(pdf_path, fig)


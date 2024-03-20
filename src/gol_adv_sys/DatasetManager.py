import numpy as np
import logging

import torch
from torch.utils.data import Dataset


from config.paths import DATASET_DIR
from src.gol_adv_sys.utils import constants as constants
from src.gol_adv_sys.utils.simulation_functions import simulate_config


class DatasetCreator():
    """
    Responsible for creating and managing a dataset. This class
    handles the generation of initial configurations, simulation to obtain final configurations,
    calculation of metrics for these configurations, and the organization of the data into
    training, validation, and test sets.

    Attributes:
        device_manager (DeviceManager): An instance of DeviceManager to manage device selection.
        dataset_train_path (str): Path to save the training dataset.
        dataset_val_path (str): Path to save the validation dataset.
        dataset_test_path (str): Path to save the test dataset.
        data (torch.Tensor): The complete dataset generated.

    """

    def __init__(self, device_manager) -> None:
        """
        Initializes the DatasetCreator with a specific device manager and predefined constants.
        Automatically generates the dataset upon initialization.

        Parameters:
            device_manager (DeviceManager): The device manager for handling device selection.
        """

        self.device_manager = device_manager

        self.__simulation_topology = constants.TOPOLOGY_TYPE["toroidal"]

        self.dataset_entire_path = DATASET_DIR / f"{constants.dataset_name}.pt"
        self.dataset_train_path  = DATASET_DIR / f"{constants.dataset_name}_train.pt"
        self.dataset_val_path    = DATASET_DIR / f"{constants.dataset_name}_val.pt"
        self.dataset_test_path   = DATASET_DIR / f"{constants.dataset_name}_test.pt"

        self.metadata_entire_path = DATASET_DIR / f"{constants.dataset_name}_metadata.pt"
        self.metadata_train_path  = DATASET_DIR / f"{constants.dataset_name}_metadata_train.pt"
        self.metadata_val_path    = DATASET_DIR / f"{constants.dataset_name}_metadata_val.pt"
        self.metadata_test_path   = DATASET_DIR / f"{constants.dataset_name}_metadata_test.pt"

        self.total_configs = 0


    def create_dataset(self) -> bool:
        """
        Generates a dataset if it does not already exist. The dataset consists of initial
        configurations and their corresponding final configurations after simulation, along with
        easy, medium, and hard metrics for each configuration. The dataset is divided into
        training, validation, and test sets.

        This method checks if the dataset already exists to avoid unnecessary computation. If the
        dataset does not exist, it proceeds with generation and saves the data to disk.

        Returns:
            torch.Tensor: The complete dataset including initial and final configurations,
            along with the metrics.

        """

        # Check if the data already exists, if not create it
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        # Check if the data folder is empty
        if len(list(DATASET_DIR.iterdir())) > 0:
            logging.warning("Data folder is not empty. Skipping dataset generation.")
            return False

        n_batches = constants.dataset_n_configs // constants.dataset_bs
        n = constants.grid_size ** 2 + 1
        batches_for_n_cells = n_batches // n
        configs = []
        metadata = []

        ids = torch.arange(constants.dataset_n_configs, dtype=torch.int32, device=self.device_manager.default_device)
        # Generate the configurations for the dataset
        for n_cells in range(n):
            logging.info(f"Generating configurations for {n_cells} living cells")
            for i in range(batches_for_n_cells):
                batch_number = i + n_cells * batches_for_n_cells
                # Initialize the batch of configurations with all cells dead (0)
                initial_config = torch.zeros(constants.dataset_bs, constants.grid_size, constants.grid_size, dtype=torch.float32,
                                             device=self.device_manager.default_device)
                # For each configuration in the batch
                for i in range(constants.dataset_bs):
                    flat_indices = torch.randperm(constants.grid_size ** 2, device=self.device_manager.default_device)[:n_cells]
                    rows, cols = flat_indices // constants.grid_size, flat_indices % constants.grid_size
                    initial_config[i, rows, cols] = 1.0
                initial_config = initial_config.view(constants.dataset_bs, 1, constants.grid_size, constants.grid_size)
                with torch.no_grad():
                    final, metrics, n_cells_init, n_cells_final = simulate_config(
                                                            config=initial_config, topology=self.__simulation_topology,
                                                            steps=constants.dataset_n_simulation_steps, calculate_final_config=True,
                                                            device=self.device_manager.default_device)
                configs.append({
                    "initial": initial_config,
                    "final": final["config"],
                    "metric_easy": metrics["easy"]["config"],
                    "metric_medium": metrics["medium"]["config"],
                    "metric_hard": metrics["hard"]["config"],
                    "metric_stable": metrics["stable"]["config"]
                })
                metadata.append({
                    "id": ids[batch_number*constants.dataset_bs: (batch_number+1)*constants.dataset_bs],
                    "n_cells_init": n_cells_init,
                    "n_cells_final": n_cells_final,
                    "transient_phase": final["transient_phase"],
                    "period": final["period"],
                    "easy_minimum": metrics["easy"]["minimum"],
                    "easy_maximum": metrics["easy"]["maximum"],
                    "easy_q1": metrics["easy"]["q1"],
                    "easy_q2": metrics["easy"]["q2"],
                    "easy_q3": metrics["easy"]["q3"],
                    "medium_minimum": metrics["medium"]["minimum"],
                    "medium_maximum": metrics["medium"]["maximum"],
                    "medium_q1": metrics["medium"]["q1"],
                    "medium_q2": metrics["medium"]["q2"],
                    "medium_q3": metrics["medium"]["q3"],
                    "hard_minimum": metrics["hard"]["minimum"],
                    "hard_maximum": metrics["hard"]["maximum"],
                    "hard_q1": metrics["hard"]["q1"],
                    "hard_q2": metrics["hard"]["q2"],
                    "hard_q3": metrics["hard"]["q3"],
                    "stable_minimum": metrics["stable"]["minimum"],
                    "stable_maximum": metrics["stable"]["maximum"],
                    "stable_q1": metrics["stable"]["q1"],
                    "stable_q2": metrics["stable"]["q2"],
                    "stable_q3": metrics["stable"]["q3"]
                })

        # Configs
        concatenated_configs_tensor = []
        configs_keys = configs[0].keys()
        for key in configs_keys:
            concatenated_tensor = torch.cat([config[key] for config in configs], dim=0)
            concatenated_configs_tensor.append(concatenated_tensor)
        configs_data = torch.stack(concatenated_configs_tensor, dim=1)

        # Metadata
        metadata_data = {}
        for key in metadata[0].keys():  # Get all keys from the first batch as reference
            # Gather the tensors for the current key from each batch
            tensors = [batch[key].to(self.device_manager.default_device) for batch in metadata]
            # Stack the tensors along a new dimension (creating a new batch dimension)
            metadata_data[key] = torch.cat(tensors, dim=0)

        # Generate shuffled indices based on the total number of configurations
        self.total_configs = len(configs_data)  # Should be constants.dataset_n_configs
        shuffled_indices = torch.randperm(self.total_configs, device=self.device_manager.default_device)

        # Apply the shuffled indices to flatten configs and metadata
        shuffled_configs  = [configs_data[i] for i in shuffled_indices]
        shuffled_metadata = {key: metadata_data[key][shuffled_indices] for key in metadata_data.keys()}

        # Save shuffled data to disk
        torch.save(shuffled_configs, self.dataset_entire_path)
        torch.save(shuffled_metadata, self.metadata_entire_path)

        return True


    def save_tensors(self):

        if self.dataset_entire_path.exists() and self.metadata_entire_path.exists():
            shuffled_configs  = torch.load(self.dataset_entire_path)
            shuffled_metadata = torch.load(self.metadata_entire_path)

        # Convert the shuffled metadata to a list of dictionaries
        metadata = [{} for _ in range(self.total_configs)]  # Initialize a list of dictionaries
        for i in range(self.total_configs):
            metadata[i] = {
                key: shuffled_metadata[key][i].item() for key in shuffled_metadata.keys()
            }

        # Convert the shuffled configs to a tensor
        data = torch.stack(shuffled_configs, dim=0)

        # Divide the data into training, validation, and test sets (use shuffled_metadata for indexing)
        n_train = int(constants.dataset_train_ratio * data.size(0))
        n_val   = int(constants.dataset_val_ratio * data.size(0))

        train_data = data[:n_train]
        val_data   = data[n_train:n_train + n_val]
        test_data  = data[n_train + n_val:]
        train_metadata = metadata[:n_train]
        val_metadata   = metadata[n_train:n_train + n_val]
        test_metadata  = metadata[n_train + n_val:]

        # Save the datasets and their corresponding metadata
        torch.save(train_data, self.dataset_train_path)
        torch.save(val_data, self.dataset_val_path)
        torch.save(test_data, self.dataset_test_path)
        torch.save(train_metadata, self.metadata_train_path)
        torch.save(val_metadata, self.metadata_val_path)
        torch.save(test_metadata, self.metadata_test_path)

        return (train_data, val_data, test_data)


class FixedDataset(Dataset):
    """
    A PyTorch Dataset class for loading data from a .pt (PyTorch saved tensor) file.

    This class provides an interface for accessing individual data points from a dataset
    stored in a .pt file, enabling compatibility with PyTorch DataLoader for batch
    processing and shuffling.

    Attributes:
        path (str): The file system path to the .pt file containing the dataset.
        data (torch.Tensor): The tensor loaded from the .pt file.

    Parameters:
        path (str): The file system path to the .pt file. This is expected to be a
                    valid path from which a PyTorch tensor can be loaded.

    Raises:
        IOError: If the file cannot be opened or read. Note that this is implicitly
                 raised when attempting to load the tensor using `torch.load`.
    """


    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self.data = torch.load(path)
        except IOError as e:
            raise e


    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.


        Returns:
            int: The size of the dataset.
        """
        return len(self.data)


    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the data point at the specified index in the dataset.

        Parameters:
            idx (int): An index into the dataset indicating which sample to retrieve.

        Returns:
            torch.Tensor: The data sample corresponding to the given index.
        """
        return self.data[idx]


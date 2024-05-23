import numpy as np
import logging

import torch
from   torch.utils.data import Dataset, DataLoader


from configs.paths import DATASET_DIR
from configs.constants import *

from src.common.utils.simulation_functions import simulate_config


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

        self.__simulation_topology = TOPOLOGY_TOROIDAL

        self.dataset_entire_path = DATASET_DIR / f"{DATASET_NAME}.pt"
        self.dataset_train_path  = DATASET_DIR / f"{DATASET_NAME}_{TRAIN}.pt"
        self.dataset_val_path    = DATASET_DIR / f"{DATASET_NAME}_{VALIDATION}.pt"
        self.dataset_test_path   = DATASET_DIR / f"{DATASET_NAME}_{TEST}.pt"

        self.metadata_entire_path = DATASET_DIR / f"{DATASET_NAME}_metadata.pt"
        self.metadata_train_path  = DATASET_DIR / f"{DATASET_NAME}_metadata_{TRAIN}.pt"
        self.metadata_val_path    = DATASET_DIR / f"{DATASET_NAME}_metadata_{VALIDATION}.pt"
        self.metadata_test_path   = DATASET_DIR / f"{DATASET_NAME}_metadata_{TEST}.pt"

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

        n_batches = DATASET_N_TOTAL_CONFIGS // DATASET_BATCH_SIZE
        n = GRID_SIZE ** 2 + 1
        batches_for_n_cells = n_batches // n
        configs = []
        metadata = []

        ids = torch.arange(DATASET_N_TOTAL_CONFIGS, dtype=torch.int32, device=self.device_manager.default_device)

        # Generate the configurations for the dataset
        for n_cells in range(n):
            logging.info(f"Generating configurations for {n_cells} living cells")
            for i in range(batches_for_n_cells):
                batch_number = i + n_cells * batches_for_n_cells
                # Initialize the batch of configurations with all cells dead (0)
                initial_config = torch.zeros(DATASET_BATCH_SIZE, GRID_SIZE, GRID_SIZE, dtype=torch.float32,
                                             device=self.device_manager.default_device)
                # For each configuration in the batch
                for i in range(DATASET_BATCH_SIZE):
                    flat_indices = torch.randperm(GRID_SIZE ** 2, device=self.device_manager.default_device)[:n_cells]
                    rows, cols = flat_indices // GRID_SIZE, flat_indices % GRID_SIZE
                    initial_config[i, rows, cols] = 1.0
                initial_config = initial_config.view(DATASET_BATCH_SIZE, 1, GRID_SIZE, GRID_SIZE)
                with torch.no_grad():
                    sim_results = simulate_config(config=initial_config, topology=self.__simulation_topology,
                                              steps=DATASET_N_SIM_STEPS, device=self.device_manager.default_device)

                    final           = sim_results["final"]
                    metrics         = sim_results["all_metrics"]
                    n_cells_initial = sim_results["n_cells_initial"]
                    n_cells_final   = sim_results["n_cells_final"]
                    period          = sim_results["period"]
                    transient_phase = sim_results["transient_phase"]

                configs.append({
                    CONFIG_INITIAL: initial_config,
                    CONFIG_FINAL: final,
                    CONFIG_TARGET_EASY: metrics[CONFIG_TARGET_EASY]["config"],
                    CONFIG_TARGET_MEDIUM: metrics[CONFIG_TARGET_MEDIUM]["config"],
                    CONFIG_TARGET_HARD: metrics[CONFIG_TARGET_HARD]["config"],
                    CONFIG_TARGET_STABLE: metrics[CONFIG_TARGET_STABLE]["config"]
                })
                metadata.append({
                    META_ID: ids[batch_number*DATASET_BATCH_SIZE: (batch_number+1)*DATASET_BATCH_SIZE],
                    META_N_CELLS_INITIAL : n_cells_initial,
                    META_N_CELLS_FINAL: n_cells_final,
                    META_TRANSIENT_PHASE: transient_phase,
                    META_PERIOD: period,
                    META_EASY_MIN: metrics[CONFIG_TARGET_EASY]["minimum"],
                    META_EASY_MAX: metrics[CONFIG_TARGET_EASY]["maximum"],
                    META_EASY_Q1: metrics[CONFIG_TARGET_EASY]["q1"],
                    META_EASY_Q2: metrics[CONFIG_TARGET_EASY]["q2"],
                    META_EASY_Q3: metrics[CONFIG_TARGET_EASY]["q3"],
                    META_MEDIUM_MIN: metrics[CONFIG_TARGET_MEDIUM]["minimum"],
                    META_MEDIUM_MAX: metrics[CONFIG_TARGET_MEDIUM]["maximum"],
                    META_MEDIUM_Q1: metrics[CONFIG_TARGET_MEDIUM]["q1"],
                    META_MEDIUM_Q2: metrics[CONFIG_TARGET_MEDIUM]["q2"],
                    META_MEDIUM_Q3: metrics[CONFIG_TARGET_MEDIUM]["q3"],
                    META_HARD_MIN: metrics[CONFIG_TARGET_HARD]["minimum"],
                    META_HARD_MAX: metrics[CONFIG_TARGET_HARD]["maximum"],
                    META_HARD_Q1: metrics[CONFIG_TARGET_HARD]["q1"],
                    META_HARD_Q2: metrics[CONFIG_TARGET_HARD]["q2"],
                    META_HARD_Q3: metrics[CONFIG_TARGET_HARD]["q3"],
                    META_STABLE_MIN: metrics[CONFIG_TARGET_STABLE]["minimum"],
                    META_STABLE_MAX: metrics[CONFIG_TARGET_STABLE]["maximum"],
                    META_STABLE_Q1: metrics[CONFIG_TARGET_STABLE]["q1"],
                    META_STABLE_Q2: metrics[CONFIG_TARGET_STABLE]["q2"],
                    META_STABLE_Q3: metrics[CONFIG_TARGET_STABLE]["q3"]
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
        self.total_configs = len(configs_data)  # Should be DATASET_N_TOTAL_CONFIGS
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
        n_train = int(DATASET_TRAIN_RATIO * data.size(0))
        n_val   = int(DATASET_VAL_RATIO * data.size(0))

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


class PairedDataset(Dataset):
    """
    A PyTorch Dataset class for loading paired data from two .pt files.

    This class provides an interface for accessing paired data points from two datasets
    stored in separate .pt files. It enables compatibility with PyTorch DataLoader for
    batch processing and shuffling.

    Attributes:
        data_set (FixedDataset): The dataset containing the data samples.
        meta_set (FixedDataset): The dataset containing the metadata samples.

    Parameters:
        data_set (FixedDataset): The dataset containing the data samples.
        meta_set (FixedDataset): The dataset containing the metadata samples.

    Raises:
        ValueError: If the data and metadata sets are not of the same size.

    """
    def __init__(self, data_set, meta_set):
        self.data_set = data_set
        self.meta_set = meta_set

        if len(self.data_set) != len(self.meta_set):
            raise ValueError(f"Data and metadata sets must be of the same size. "
                             f"Data set size: {len(self.data_set)}, "
                             f"Metadata set size: {len(self.meta_set)}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data_set)


    def __getitem__(self, idx):
        """
        Retrieves the data and metadata points at the specified index in the dataset.

        Parameters:
            idx (int): An index into the dataset indicating which sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the data and metadata samples

        """
        data = self.data_set[idx]
        meta = self.meta_set[idx]
        return data, meta


class DatasetManager():
    def __init__(self) -> None:

        self.dataset_types = [TRAIN, VALIDATION, TEST]


    def get_dataloader(self,
                       dataset_type: str,
                       batch_size: int,
                       shuffle: bool = True) -> DataLoader:
        """
        Returns the DataLoader for the specified dataset.

        Parameters:
            dataset_type (str): The type of the dataset to retrieve the DataLoader for.

        Returns:
            DataLoader: The DataLoader for the specified dataset.
        """

        if dataset_type not in self.dataset_types:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        paired_dataset = self.get_dataset(dataset_type)

        return DataLoader(paired_dataset, batch_size=batch_size, shuffle=shuffle)


    def get_dataset(self, dataset_type: str) -> PairedDataset:
        """
        Returns a specified dataset containing data and metadata.

        Parameters:
            dataset_type (str): The type of the dataset.

        """
        if dataset_type not in self.dataset_types:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        fixed_data = FixedDataset(f"{DATASET_DIR}/{DATASET_NAME}_{dataset_type}.pt")
        fixed_meta = FixedDataset(f"{DATASET_DIR}/{DATASET_NAME}_metadata_{dataset_type}.pt")

        return PairedDataset(fixed_data, fixed_meta)


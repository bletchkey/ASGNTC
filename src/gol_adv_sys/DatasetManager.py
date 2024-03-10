import os
import datetime

import torch
from torch.utils.data import Dataset

from .utils import constants as constants

from .utils.simulation_functions import simulate_config


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

        self.dataset_train_path = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_train.pt"))
        self.dataset_val_path = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_val.pt"))
        self.dataset_test_path = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_test.pt"))

        self.data = self.create_fixed_dataset()


    def create_fixed_dataset(self):
        """
        Generates a fixed dataset if it does not already exist. The dataset consists of initial
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
        if os.path.exists(constants.fixed_dataset_path):
            pass
        else:
            os.makedirs(constants.fixed_dataset_path)

        # Check if the data folder is empty
        if len(os.listdir(constants.fixed_dataset_path)) > 0:
            print("data folder is not empty")

        else:

            data = None
            n_batches = constants.fixed_dataset_n_configs // constants.fixed_dataset_bs
            n = constants.grid_size ** 2 + 1
            batches_for_n_cells = n_batches // n
            configs = []

            # Generate the configurations for the fixed dataset
            for n_cells in range(n):
                print(f"Generating configurations for n_cells = {n_cells}")
                for _ in range(batches_for_n_cells):
                    # Initialize the batch of configurations with all cells dead (0)
                    initial_config = torch.zeros(constants.fixed_dataset_bs, constants.grid_size, constants.grid_size, dtype=torch.float32,
                                                 device=self.device_manager.default_device)

                    # For each configuration in the batch
                    for i in range(constants.fixed_dataset_bs):
                        flat_indices = torch.randperm(constants.grid_size ** 2, device=self.device_manager.default_device)[:n_cells]
                        rows, cols = flat_indices // constants.grid_size, flat_indices % constants.grid_size
                        initial_config[i, rows, cols] = 1.0

                    initial_config = initial_config.view(constants.fixed_dataset_bs, 1, constants.grid_size, constants.grid_size)

                    with torch.no_grad():
                        final_config, metrics = simulate_config(config=initial_config, topology=self.__simulation_topology,
                                                                steps=constants.fixed_dataset_n_simulation_steps, calculate_final_config=True,
                                                                device=self.device_manager.default_device)

                    configs.append({
                        "initial": initial_config,
                        "final": final_config,
                        "metric_easy": metrics["easy"],
                        "metric_medium": metrics["medium"],
                        "metric_hard": metrics["hard"],
                    })

            concatenated_tensors = []
            keys = configs[0].keys()

            for key in keys:
                concatenated_tensor = torch.cat([config[key] for config in configs], dim=0)
                concatenated_tensors.append(concatenated_tensor)

            # Stack the tensors
            data = torch.stack(concatenated_tensors, dim=1)

            # Shuffle the data
            data = data[torch.randperm(data.size(0))]

            # Divide the data into training, validation and test sets
            n_train = int(constants.fixed_dataset_train_ratio * data.size(0))
            n_val   = int(constants.fixed_dataset_val_ratio * data.size(0))
            train_data = data[:n_train]
            val_data   = data[n_train:n_train+n_val]
            test_data  = data[n_train+n_val:]

            # Save the data sets
            torch.save(train_data, self.dataset_train_path)
            torch.save(val_data, self.dataset_val_path)
            torch.save(test_data, self.dataset_test_path)

            return (train_data, val_data, test_data)

        return None


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


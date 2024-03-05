import os

import torch
from torch.utils.data import Dataset

from .utils import constants as constants


from .FolderManager import FolderManager

from .utils.helper_functions import generate_new_batches_fixed_dataset, check_train_fixed_dataset_configs
from .utils.simulation_functions import simulate_config_fixed_dataset


class DatasetCreator():
    def __init__(self, device_manager) -> None:

        self.folders = FolderManager()
        self.device_manager = device_manager

        self.simulation_topology = constants.TOPOLOGY_TYPE["toroidal"]

        self.dataset_train_path = os.path.join(self.folders.data_folder, str(constants.fixed_dataset_name+"_train.pt"))
        self.dataset_val_path = os.path.join(self.folders.data_folder, str(constants.fixed_dataset_name+"_val.pt"))
        self.dataset_test_path = os.path.join(self.folders.data_folder, str(constants.fixed_dataset_name+"_test.pt"))

        self.data = self.create_fixed_dataset()


    def create_fixed_dataset(self):

        if len(os.listdir(self.__data_folder)) != 0:
            print("data already exists and it's not empty")

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
                    final_config, metrics = simulate_config_fixed_dataset(initial_config, self.simulation_topology,
                                                                      constants.fixed_dataset_metric_steps,
                                                                      self.device_manager.default_device)

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


class FixedDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


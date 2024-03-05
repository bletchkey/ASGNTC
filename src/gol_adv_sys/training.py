"""
This module contains the class Training.
This class is used to train the generator and predictor models.

The training is done using the adversarial training approach.
"""

import numpy as np

import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import datetime

from .utils import constants as constants

from .Predictor import Predictor_Baseline
from .Generator import Generator

from .FolderManager import FolderManager
from .DeviceManager import DeviceManager
from .DatasetManager import FixedDataset

from .utils.helper_functions import test_models, save_progress_plot, save_losses_plot, test_predictor_model
from .utils.helper_functions import generate_new_batches, get_data_tensor, get_elapsed_time_str, get_config_from_batch


class Training():
    def __init__(self) -> None:
        self.__date = datetime.datetime.now()

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]
        self.__set_seed()

        self.folders = FolderManager(self.__date)
        self.device_manager = DeviceManager()

        # Get rid of the following two lines
        self.simulation_topology = constants.TOPOLOGY_TYPE["toroidal"]
        self.init_config_type = constants.INIT_CONFIG_TYPE["threshold"]


        self.metric_type = constants.METRIC_TYPE["medium"]

        self.current_epoch = 0
        self.step_times_secs = []

        self.losses = {"generator": [],
                       "predictor_train": [],
                       "predictor_val": [],
                       "predictor_test": [],}

        self.lr_each_epoch = {
            "predictor": [],
            "generator": [],
        }

        self.n_times_trained_p = 0
        self.n_times_trained_g = 0

        self.criterion_p = nn.MSELoss()
        self.criterion_g = lambda x, y: -self.criterion_p(x, y)

        self.model_p = Predictor_Baseline().to(self.device_manager.default_device)
        self.model_g = Generator(noise_std=0).to(self.device_manager.default_device)

        self.optimizer_p = optim.SGD(self.model_p.parameters(),
                                     lr=constants.p_sgd_lr,
                                     momentum=constants.p_sgd_momentum,
                                     weight_decay=constants.p_sgd_wd)

        self.optimizer_g = optim.AdamW(self.model_g.parameters(),
                                       lr=constants.g_adamw_lr,
                                       betas=(constants.g_adamw_b1, constants.g_adamw_b2),
                                       eps=constants.g_adamw_eps,
                                       weight_decay=constants.g_adamw_wd)

        self.fixed_noise = torch.randn(constants.bs, constants.nz, 1, 1, device=self.device_manager.default_device)

        self.properties_g= {"enabled": True, "can_train": False}

        self.load_models = {"predictor": False, "generator": False, "name_p": None, "name_g": None}

        self.train_dataloader = []
        self.test_dataloader = []
        self.val_dataloader = []

        self.data_tensor = None

        self.fixed_dataset = {"enabled": True, "train_data": None, "val_data": None, "test_data": None}

        self.path_log_file = self.__init_log_file() if self.fixed_dataset["enabled"] == False else self.__init_log_file_fixed_dataset()
        self.path_p        = None
        self.path_g        = None


    """
    Function to run the training.

    """
    def run(self):

        if self.fixed_dataset["enabled"]:

            train_path = os.path.join(self.folders.data_folder, "gol_fixed_dataset_train.pt")
            val_path   = os.path.join(self.folders.data_folder, "gol_fixed_dataset_val.pt")
            test_path  = os.path.join(self.folders.data_folder, "gol_fixed_dataset_test.pt")

            self.fixed_dataset["train_data"] = FixedDataset(train_path)
            self.fixed_dataset["val_data"]   = FixedDataset(val_path)
            self.fixed_dataset["test_data"]  = FixedDataset(test_path)

            self.train_dataloader = DataLoader(self.fixed_dataset["train_data"], batch_size=constants.bs, shuffle=True)
            self.val_dataloader   = DataLoader(self.fixed_dataset["val_data"], batch_size=constants.bs, shuffle=True)
            self.test_dataloader  = DataLoader(self.fixed_dataset["test_data"], batch_size=constants.bs, shuffle=True)

            self.__fit_p_on_fixed_dataset()
        else:

            self.__check_load_models(self.load_models["name_p"], self.load_models["name_g"])
            self.__fit()


    """
    Training loop for the predictor model.

    The predictor model is trained on the fixed data set.

    """
    def __fit_p_on_fixed_dataset(self):

        torch.autograd.set_detect_anomaly(True)

        if self.device_manager.n_balanced_gpus > 0:
            self.model_p = nn.DataParallel(self.model_p, device_ids=self.device_manager.balanced_gpu_indices)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_p, mode="min", factor=0.1, patience=2, verbose=True,
                                                         threshold=1e-4, threshold_mode="rel", cooldown=2, min_lr=0, eps=1e-8)

        # Training loop
        for epoch in range(constants.num_epochs):
            self.current_epoch = epoch
            self.lr_each_epoch["predictor"].append(self.__get_lr(self.optimizer_p))

            epoch_start_time = time.time()

            # Train the predictor model
            train_loss = 0
            self.model_p.train()
            for batch in self.train_dataloader:
                self.optimizer_p.zero_grad()
                predicted_metric = self.model_p(self.__get_initial_config(batch))
                errP = self.criterion_p(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                errP.backward()
                self.optimizer_p.step()
                train_loss += errP.item()

            self.n_times_trained_p += 1
            train_loss /= len(self.train_dataloader)
            self.losses["predictor_train"].append(train_loss)

            # Check the validation loss
            val_loss = 0
            self.model_p.eval()
            with torch.no_grad():
                for batch in self.val_dataloader:
                    predicted_metric = self.model_p(self.__get_initial_config(batch))
                    errP = self.criterion_p(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                    val_loss += errP.item()

            val_loss /= len(self.val_dataloader)
            self.losses["predictor_val"].append(val_loss)

            # Update the learning rate
            scheduler.step(val_loss)

            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time

            # Log the training epoch progress
            self.__log_training_epoch_p(epoch_elapsed_time)

            # Test and save models
            data = self.__test_predictor_model()
            self.__save_progress_plot(data)
            self.__save_losses_plot()
            self.__save_models()
            self.__resource_cleanup

        # Check the test loss
        test_loss = 0
        self.model_p.eval()
        with torch.no_grad():
            for batch in self.test_dataloader:
                predicted_metric = self.model_p(self.__get_initial_config(batch))
                errP = self.criterion_p(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                test_loss += errP.item()

        test_loss /= len(self.test_dataloader)

        self.losses["predictor_test"].append(test_loss)
        str_err_p_test = f"{self.losses['predictor_test'][-1]}"

        with open(self.path_log_file, "a") as log:
            log.write("\n\nPerformance of the predictor model on the test set:\n")
            log.write(f"Loss P (test): {str_err_p_test}\n\n")
            log.write(f"\n\nTraining ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            log.write(f"Number of times P was trained: {self.n_times_trained_p}\n")
            log.flush()


    """
    Adversarial training loop.

    """
    def __fit(self):

        torch.autograd.set_detect_anomaly(True)

        if self.device_manager.n_balanced_gpus > 0:
            self.model_p = nn.DataParallel(self.model_p, device_ids=self.device_manager.balanced_gpu_indices)
            self.model_g = nn.DataParallel(self.model_g, device_ids=self.device_manager.balanced_gpu_indices)

        for epoch in range(constants.num_epochs):

            self.step_times_secs.append([])
            self.current_epoch = epoch

            self.__get_train_dataloader()

            with open(self.path_log_file, "a") as log:

                log.write(f"\nEpoch: {epoch+1}/{constants.num_epochs}\n")
                log.write(f"Number of generated configurations in the dataset: {len(self.train_dataloader)*constants.bs}\n\n")
                log.flush()

            for step in range(constants.num_training_steps):

                step_start_time = time.time()

                self.__train_predictor()

                if self.properties_g["enabled"]:
                    self.__train_generator()

                step_end_time = time.time()
                self.step_times_secs[self.current_epoch].append(step_end_time - step_start_time)

                self.__log_training_step(step)

            with open(self.path_log_file, "a") as log:
                log.write(f"\nElapsed time: {get_elapsed_time_str(self.step_times_secs[epoch])}\n")
                if self.n_times_trained_p > 0:
                    log.write(f"Average loss P: {self.__get_loss_avg_p_last_epoch()}\n")
                if self.n_times_trained_g > 0:
                    log.write(f"Average loss G: {self.__get_loss_avg_g_last_epoch()}\n")
                log.flush()

            # Update properties for G
            self.__can_g_train()

            # Test and save models
            data = self.__test_models()
            self.__save_progress_plot(data)
            self.__save_models()

            self.__resource_cleanup

        with open(self.path_log_file, "a") as log:
            log.write(f"\n\nTraining ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            log.write(f"Number of times P was trained: {self.n_times_trained_p}\n")
            log.write(f"Number of times G was trained: {self.n_times_trained_g}\n")
            log.flush()


    """
    Log the progress of the training session inside each epoch for the predictor model.

    """
    def __log_training_epoch_p(self, time):
        with open(self.path_log_file, "a") as log:

            str_epoch_time  = f"{get_elapsed_time_str(time)}"
            str_epoch       = f"{self.current_epoch+1}/{constants.num_epochs}"
            str_err_p_train = f"{self.losses['predictor_train'][-1]}"
            str_err_p_val   = f"{self.losses['predictor_val'][-1]}"

            lr = self.lr_each_epoch["predictor"][self.current_epoch]

            log.write(f"{str_epoch_time} | Epoch: {str_epoch}, Loss P (train): {str_err_p_train}, Loss P (val): {str_err_p_val}, [LR: {lr}]\n")
            log.flush()


    """
    Log the progress of the training session inside each epoch.

    """
    def __log_training_step(self, step):
        with open(self.path_log_file, "a") as log:

            str_step_time = f"{get_elapsed_time_str(self.step_times_secs[self.current_epoch][step])}"
            str_step      = f"{step+1}/{constants.num_training_steps}"
            str_err_p     = f"{self.losses['predictor_train'][-1]}"
            str_err_g     = f"{self.losses['generator'][-1]}" if len(self.losses['generator']) > 0 else "N/A"

            log.write(f"{str_step_time} | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}\n")
            log.flush()


    """
    Create a log file for the training session.
    When creating the log file, the training specifications are also written to the file.

    Returns:
        path (str): The path to the log file.

    """
    def __init_log_file(self):

        path = os.path.join(self.folders.logs_folder, "asgntc.txt")

        with open(path, "w") as log_file:
            log_file.write(f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n\n")

            if self.__seed_type["random"]:
                log_file.write(f"Random seed: {self.seed}\n")
            elif self.__seed_type["fixed"]:
                log_file.write(f"Fixed seed: {self.seed}\n")
            else:
                log_file.write(f"Unknown seed\n")

            log_file.write(f"Default device: {self.device_manager.default_device}\n")
            if len(self.device_manager.balanced_gpu_indices) > 0:
                log_file.write(f"Balanced GPU indices: {self.device_manager.balanced_gpu_indices}\n")

            log_file.write(f"\nTraining specs:\n")
            log_file.write(f"Batch size: {constants.bs}\n")
            log_file.write(f"Epochs: {constants.num_epochs}\n")
            log_file.write(f"Number of training steps in each epoch: {constants.num_training_steps}\n")

            log_file.write(f"Number of batches generated in each epoch: {constants.n_batches} ({constants.n_configs} configs)\n")
            log_file.write(f"Max number of generated batches in data set: {constants.n_max_batches} ({constants.n_max_configs} configs)\n")

            log_file.write(f"\nSimulation specs:\n")
            log_file.write(f"Grid size: {constants.grid_size}\n")
            log_file.write(f"Simulation steps: {constants.n_simulation_steps}\n")

            if self.simulation_topology == constants.TOPOLOGY_TYPE["toroidal"]:
                log_file.write(f"Topology: toroidal\n")
            elif self.simulation_topology == constants.TOPOLOGY_TYPE["flat"]:
                log_file.write(f"Topology: flat\n")
            else:
                log_file.write(f"Topology: unknown\n")

            if self.init_config_type == constants.INIT_CONFIG_TYPE["threshold"]:
                log_file.write(f"Initial configuration type: threshold\n")
                log_file.write(f"Threshold for the value of the cells: {constants.threshold_cell_value}\n")
            elif self.init_config_type == constants.INIT_CONFIG_TYPE["n_living_cells"]:
                log_file.write(f"Initial configuration type: n_living_cells\n")
                log_file.write(f"Number of living cells in initial grid: {constants.n_living_cells}\n")
            else:
                log_file.write(f"Initial configuration type: unknown\n")


            log_file.write(f"\nModel specs:\n")
            log_file.write(f"Optimizer P: {self.optimizer_p.__class__.__name__}\n")
            log_file.write(f"Criterion P: {self.criterion_p.__class__.__name__}\n")

            if self.load_models["predictor"]:
                log_file.write(f"The predictor model has been loaded from a previous training session\n")

            if self.properties_g["enabled"]:
                if self.load_models["generator"]:
                    log_file.write(f"The generator model has been loaded from a previous training session\n")
                log_file.write(f"Latent space size: {constants.nz}\n")
                log_file.write(f"Optimizer G: {self.optimizer_g.__class__.__name__}\n")
                log_file.write(f"Criterion G: -{self.criterion_p.__class__.__name__}\n")

                if self.properties_g["can_train"]:
                    log_file.write(f"\nThe generator model starts to train at the beginning of the training session\n")
                else:
                    log_file.write(f"\nAverage loss of P before training G: {constants.threshold_avg_loss_p}\n\n")

            log_file.write(f"\n\nTraining progress:\n\n")

            log_file.flush()

        return path


    """
    Create a log file for the training session on the fixed dataset.
    When creating the log file, the training specifications are also written to the file.

    Returns:
        path (str): The path to the log file.

    """
    def __init_log_file_fixed_dataset(self):

        path = os.path.join(self.folders.logs_folder, "asgntc.txt")

        with open(path, "w") as log_file:
            log_file.write(f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n\n")

            if self.__seed_type["random"]:
                log_file.write(f"Random seed: {self.seed}\n")
            elif self.__seed_type["fixed"]:
                log_file.write(f"Fixed seed: {self.seed}\n")
            else:
                log_file.write(f"Unknown seed\n")

            log_file.write(f"Default device: {self.device_manager.default_device}\n")
            if len(self.device_manager.balanced_gpu_indices) > 0:
                log_file.write(f"Balanced GPU indices: {self.device_manager.balanced_gpu_indices}\n")

            log_file.write(f"\nTraining specs:\n")
            log_file.write(f"Batch size: {constants.bs}\n")
            log_file.write(f"Epochs: {constants.num_epochs}\n")

            log_file.write(f"\n\nTraining progress:\n\n")

            log_file.flush()

        return path

    """
    Get the dataloader for the current epoch.

    Each epoch, a new dataloader is created by adding n_batches new configurations to the dataloader, for a total of
    n_configs new configurations.
    The maximum number of batches in the dataloader is n_max_batches, that contains n_max_configs configurations.
    The older batches of configurations are removed to make room for the new ones.

    The configurations are generated by the generator model.

    Returns:
        dataloader (torch.utils.data.DataLoader): The dataloader for the current epoch.

    """
    def __get_train_dataloader(self):
        data = get_data_tensor(self.data_tensor, self.model_g,
                              self.simulation_topology, self.init_config_type, self.device_manager.default_device)

        self.data_tensor = data

        # Create the dataloader from the tensor
        self.train_dataloader = DataLoader(self.data_tensor, batch_size=constants.bs, shuffle=True)

        return self.train_dataloader

    """
    Generate new configurations using the generator model.

    Args:
        n_batches (int): The number of batches to generate.

    Returns:
        new_configs (list): A list of dictionaries containing information about the generated configurations.

    """
    def __get_new_batches(self, n_batches):
        return  generate_new_batches(self.model_g, n_batches, self.simulation_topology,
                                     self.init_config_type, self.device_manager.default_device)


    """
    Generate one new batch of configurations using the generator model.

    Returns:
        new_config (dict): A dictionary containing information about the generated configurations.

    """
    def __get_one_new_batch(self):
        return self.__get_new_batches(1)


    """
    Function to get a batch of initial configurations from the batch.

    """
    def __get_initial_config(self, batch):
        return get_config_from_batch(batch, constants.CONFIG_NAMES["initial"], self.device_manager.default_device)


    """
    Function to get a batch of the specified metric type from the batch.

    """
    def __get_metric_config(self, batch, metric_type):
        return get_config_from_batch(batch, metric_type, self.device_manager.default_device)


    """
    Function for training the predictor model.

    Returns:
        loss (float): The loss of the predictor model.

    """
    def __train_predictor(self):

        loss = 0
        self.model_p.train()

        for batch in self.train_dataloader:
            self.optimizer_p.zero_grad()
            predicted_metric = self.model_p(self.__get_initial_config(batch))
            errP = self.criterion_p(predicted_metric, self.__get_metric_config(batch, self.metric_type))
            errP.backward()
            self.optimizer_p.step()
            loss += errP.item()

        self.n_times_trained_p += 1

        loss /= len(self.train_dataloader)
        self.losses["predictor_train"].append(loss)

        return loss


    """
    Function for training the generator model.

    Returns:
        loss (float): The loss of the generator model.

    """
    def __train_generator(self):

        if self.properties_g["can_train"]:

            loss = 0
            n = constants.n_batches

            self.model_g.train()
            for _ in range(n):
                batch = self.__get_one_new_batch()
                self.optimizer_g.zero_grad()
                predicted_metric = self.model_p(self.__get_initial_config(batch))
                errG = self.criterion_g(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                errG.backward()
                self.optimizer_g.step()
                loss += (-1 * errG.item())

            self.n_times_trained_g += 1

            loss /= n
            self.losses["generator"].append(loss)

            return loss

        else:
            return None


    """
    Function for getting the average loss of the predictor model.
    The average loss is calculated on the last n losses.

    If the number of losses is less than n, the average is calculated on all the losses.
    To get the average loss of the last epoch, n should be set to num_training_steps (steps per epoch).

    Args:
        on_last_n_losses (int): The number of losses to calculate the average on starting from the last loss.

    Returns:
        avg_loss_p (float): The average loss of the predictor model on the last n losses.

    """
    def __get_loss_avg_p(self, on_last_n_losses):

        len_losses_p_train = len(self.losses["predictor_train"])
        if len_losses_p_train <= 0:
            return None

        if on_last_n_losses > len_losses_p_train:
            on_last_n_losses = len_losses_p_train

        avg_loss_p = sum(self.losses["predictor_train"][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_p


    """
    Function for getting the average loss of the generator model.
    The average loss is calculated on the last n losses.

    If the number of losses is less than n, the average is calculated on all the losses.
    To get the average loss of the last epoch, n should be set to num_training_steps (steps per epoch).

    Args:
        on_last_n_losses (int): The number of losses to calculate the average on starting from the last loss.

    Returns:
        avg_loss_g (float): The average loss of the generator model on the last n losses.

    """
    def __get_loss_avg_g(self, on_last_n_losses):

        len_losses_g = len(self.losses["generator"])
        if len_losses_g <= 0:
            return None

        if on_last_n_losses > len_losses_g:
            on_last_n_losses = len_losses_g

        avg_loss_g = sum(self.losses["generator"][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_g


    """
    Special case of __get_loss_avg_p() where the average is calculated for the last epoch.

    Returns:
        avg_loss_p_last_epoch (float): The average loss of the predictor model on the last epoch.

    """
    def __get_loss_avg_p_last_epoch(self):
        return self.__get_loss_avg_p(constants.num_training_steps)


    """
    Special case of __get_loss_avg_g() where the average is calculated for the last epoch.

    Returns:
        avg_loss_g_last_epoch (float): The average loss of the generator model on the last epoch.

    """
    def __get_loss_avg_g_last_epoch(self):
        return self.__get_loss_avg_g(constants.num_training_steps)


    """
    Function for testing the models.
    The models are tested on the fixed noise.

    Returns:
        data (dict): Contains the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

    """
    def __test_models(self):
        return test_models(self.model_g, self.model_p, self.simulation_topology,
                           self.init_config_type, self.fixed_noise, self.metric_type, self.device_manager.default_device)


    """
    Function for testing the predictor model.

    Returns:
        data (dict): Contains the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

    """
    def __test_predictor_model(self):
        return test_predictor_model(self.test_dataloader, self.metric_type, self.model_p, self.device_manager.default_device)


    """
    Function for saving the progress plot.
    It save the plot that shows the generated configurations, initial configurations, simulated configurations,
    simulated metrics and predicted metrics.

    Args:
        data (dict): Contains the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

    """
    def __save_progress_plot(self, data):
        save_progress_plot(data, self.current_epoch, self.folders.results_folder)


    """
    Function
    """
    def __save_losses_plot(self):
        save_losses_plot(self.losses["predictor_train"], self.losses["predictor_val"],
                         self.lr_each_epoch["predictor"], self.folders.base_folder)


    """
    Function for saving the models.

    It saves the generator and predictor models to the models folder every n times they are trained.

    """
    def __save_models(self):

        n = 10
        epoch = self.current_epoch + 1

        if self.n_times_trained_p > 0 and self.n_times_trained_p % n == 0:
            self.path_p = os.path.join(self.folders.models_folder, f"predictor_{epoch}.pth.tar")

            if isinstance(self.model_p, nn.DataParallel):
                torch.save({
                        "state_dict": self.model_p.module.state_dict(),
                        "optimizer": self.optimizer_p.state_dict(),
                    }, self.path_p)
            else:
                torch.save({
                        "state_dict": self.model_p.state_dict(),
                        "optimizer": self.optimizer_p.state_dict(),
                    }, self.path_p)


        if self.n_times_trained_g > 0 and self.n_times_trained_g % n == 0:
            self.path_g = os.path.join(self.folders.models_folder, f"generator_{epoch}.pth.tar")

            if isinstance(self.model_g, nn.DataParallel):
                torch.save({
                        "state_dict": self.model_g.module.state_dict(),
                        "optimizer": self.optimizer_g.state_dict(),
                       }, self.path_g)
            else:
                torch.save({
                        "state_dict": self.model_g.state_dict(),
                        "optimizer": self.optimizer_g.state_dict(),
                       }, self.path_g)


    """
    Function for loading the predictor model.

    """
    def __load_predictor(self, name_p):
        path = os.path.join(constants.trained_models_path, name_p)

        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model_p.load_state_dict(checkpoint["state_dict"])
            self.optimizer_p.load_state_dict(checkpoint["optimizer"])


    """
    Function for loading the generator model.

    """
    def __load_generator(self, name_g):
        path = os.path.join(constants.trained_models_path, name_g)

        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model_g.load_state_dict(checkpoint["state_dict"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer"])


    """
    Function for checking if the models should be loaded from a previous training session.

    """
    def __check_load_models(self, name_p, name_g):
        if self.load_models["predictor"]:
            self.__load_predictor(name_p)

        if self.load_models["generator"]:
            self.__load_generator(name_g)
            self.properties_g["enabled"] = True
            self.properties_g["can_train"] = True


    """
    Function that returns if the generator model can be trained.
    The generator model can be trained if the average loss of the predictor model is less than a certain threshold.

    Returns:
        can_train (bool): True if the generator model can be trained, False otherwise.

    """
    def __can_g_train(self):

        if not self.properties_g["enabled"]:
            return False

        if self.properties_g["enabled"] and self.properties_g["can_train"]:
            return True

        if self.properties_g["enabled"] and not self.properties_g["can_train"]:
            self.properties_g["can_train"] = self.__get_loss_avg_p_last_epoch() < constants.threshold_avg_loss_p

        return self.properties_g["can_train"]


    """
    Function for retrieving the learning rate of the optimizer

    """
    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]


    """
    Function for cleaning up the resources used by the training session.
    If the device used for training is a GPU, the CUDA cache is cleared.

    """
    def __resource_cleanup(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


    """
    Function for setting the seed for the random number generators.

    """
    def __set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)


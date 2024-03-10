"""
This module contains the TrainingPredictor class.

This class is used to train the predictor model on the fixed dataset.

"""

import numpy as np

import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import datetime

from .utils import constants as constants

from .FolderManager import FolderManager
from .DeviceManager import DeviceManager
from .DatasetManager import FixedDataset

from .utils.helper_functions import save_progress_plot, save_losses_plot, test_predictor_model, get_elapsed_time_str, get_config_from_batch

from .TrainingBase import TrainingBase

class TrainingPredictor(TrainingBase):
    """
    Class designed to handle the training of the predictor model on the fixed dataset.

    Attributes:
        __date (datetime): The date and time when the training session was started.
        __seed_type (dict): The type of seed used for the random number generators.
        seed (int): The seed used for the random number generators.
        folders (FolderManager): The folder manager used for managing the folders used in the training session.
        device_manager (DeviceManager): The device manager used for managing the devices used in the training session.
        simulation_topology (str): The topology used for the simulation.
        init_config_type (str): The type of initial configuration used for the simulation.
        metric_type (str): The type of metric used for the simulation.
        current_epoch (int): The current epoch of the training session.
        step_times_secs (list): The times in seconds for each step of the training session.
        losses (dict): The losses of the predictor model during the training session.
        learning_rates (list): The learning rates used during the training session.
        n_times_trained_p (int): The number of times the predictor model was trained.
        criterion_p (nn.Module): The criterion used for the predictor model.
        model_p (nn.Module): The predictor model used for the training session.
        optimizer_p (optim.Optimizer): The optimizer used for the predictor model.
        train_dataloader (DataLoader): The dataloader used for training the predictor model.
        test_dataloader (DataLoader): The dataloader used for testing the predictor model.
        val_dataloader (DataLoader): The dataloader used for validating the predictor model.
        data_tensor (torch.Tensor): The data tensor used for the simulation.
        fixed_dataset (dict): The fixed dataset used for the training session.
        path_log_file (str): The path to the log file used for the training session.
        path_p (str): The path to the saved predictor model.

    """

    def __init__(self, model_p=None) -> None:

        self.__date = datetime.datetime.now()

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]
        self.__set_seed()

        self.folders = FolderManager(self.__date)
        self.device_manager = DeviceManager()

        self.simulation_topology = constants.TOPOLOGY_TYPE["toroidal"]
        self.init_config_type = constants.INIT_CONFIG_TYPE["threshold"]

        self.metric_type = constants.METRIC_TYPE["medium"]

        self.current_epoch = 0
        self.step_times_secs = []

        self.losses = {"predictor_train": [],
                       "predictor_val": [],
                       "predictor_test": [],}

        self.learning_rates = []

        self.n_times_trained_p = 0

        self.criterion_p = nn.MSELoss()

        self.model_p = model_p

        self.optimizer_p = optim.SGD(self.model_p.parameters(),
                                     lr=constants.p_sgd_lr,
                                     momentum=constants.p_sgd_momentum,
                                     weight_decay=constants.p_sgd_wd)

        self.train_dataloader = []
        self.test_dataloader = []
        self.val_dataloader = []

        self.data_tensor = None

        self.fixed_dataset = {"train_data": None, "val_data": None, "test_data": None}

        self.path_log_file = self.__init_log_file_fixed_dataset()
        self.path_p        = None


    def run(self):
        """
        Function used for running the training session.

        """

        self.model_p = self.model_p.to(self.device_manager.default_device)

        train_path = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_train.pt"))
        val_path   = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_val.pt"))
        test_path  = os.path.join(constants.fixed_dataset_path, str(constants.fixed_dataset_name+"_test.pt"))

        self.fixed_dataset["train_data"] = FixedDataset(train_path)
        self.fixed_dataset["val_data"]   = FixedDataset(val_path)
        self.fixed_dataset["test_data"]  = FixedDataset(test_path)

        self.train_dataloader = DataLoader(self.fixed_dataset["train_data"], batch_size=constants.bs, shuffle=True)
        self.val_dataloader   = DataLoader(self.fixed_dataset["val_data"], batch_size=constants.bs, shuffle=True)
        self.test_dataloader  = DataLoader(self.fixed_dataset["test_data"], batch_size=constants.bs, shuffle=True)

        self._fit()


    def load_predictor(self, name_p):
        """
        Function for loading the predictor model.

        Args:
            name_p (str): The name of the predictor model to load.

        """

        path = os.path.join(constants.trained_models_path, name_p)

        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model_p.load_state_dict(checkpoint["state_dict"])
            self.optimizer_p.load_state_dict(checkpoint["optimizer"])


    def _fit(self):
        """
        Training loop for the predictor model.

        The predictor model is trained on the fixed data set.

        """

        torch.autograd.set_detect_anomaly(True)

        if self.device_manager.n_balanced_gpus > 0:
            self.model_p = nn.DataParallel(self.model_p, device_ids=self.device_manager.balanced_gpu_indices)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_p, mode="min", factor=0.1, patience=2, verbose=True,
                                                         threshold=1e-4, threshold_mode="rel", cooldown=2, min_lr=0, eps=1e-8)

        # Training loop
        for epoch in range(constants.num_epochs):
            self.current_epoch = epoch
            self.learning_rates.append(self.__get_learning_rate(self.optimizer_p))

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
            self.__log_training_epoch(epoch_elapsed_time)

            # Test and save models
            data = self.__test_predictor_model()
            self.__save_progress_plot(data)
            self.__save_losses_plot()
            self.__save_predictor_model()
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


    def __log_training_epoch(self, time):
        """
        Log the progress of the training session inside each epoch for the predictor model.

        """
        with open(self.path_log_file, "a") as log:

            str_epoch_time  = f"{get_elapsed_time_str(time)}"
            str_epoch       = f"{self.current_epoch+1}/{constants.num_epochs}"
            str_err_p_train = f"{self.losses['predictor_train'][-1]}"
            str_err_p_val   = f"{self.losses['predictor_val'][-1]}"

            lr = self.learning_rates[self.current_epoch]

            log.write(f"{str_epoch_time} | Epoch: {str_epoch}, Loss P (train): {str_err_p_train}, Loss P (val): {str_err_p_val}, [LR: {lr}]\n")
            log.flush()


    def __init_log_file_fixed_dataset(self):
        """
        Create a log file for the training session on the fixed dataset.
        When creating the log file, the training specifications are also written to the file.

        Returns:
            path (str): The path to the log file.

        """

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

            log_file.write(f"Predicting metric type: {self.metric_type}\n")

            log_file.write(f"\n\nTraining progress:\n\n")

            log_file.flush()

        return path


    def __get_initial_config(self, batch):
        """
        Function to get a batch of initial configurations from the batch.

        """

        return get_config_from_batch(batch, constants.CONFIG_NAMES["initial"], self.device_manager.default_device)


    def __get_metric_config(self, batch, metric_type):
        """
        Function to get a batch of the specified metric type from the batch.

        """

        return get_config_from_batch(batch, metric_type, self.device_manager.default_device)


    def __test_predictor_model(self):
        """
        Function for testing the predictor model.

        Returns:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated metrics and predicted metrics.

        """

        return test_predictor_model(self.test_dataloader, self.metric_type, self.model_p, self.device_manager.default_device)


    def __save_progress_plot(self, data):
        """
        Function for saving the progress plot.
        It save the plot that shows the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

        Args:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated metrics and predicted metrics.

        """
        save_progress_plot(data, self.current_epoch, self.folders.results_folder)


    def __save_losses_plot(self):
        """
        Function for plotting and saving the losses of training and validation for the predictor model.

        """

        save_losses_plot(self.losses["predictor_train"], self.losses["predictor_val"],
                         self.learning_rates, self.folders.base_folder)


    def __save_predictor_model(self):
        """
        Function for saving the predictor model.

        It saves predictor model to the models folder every n epochs.

        """

        n = 10
        epoch = self.current_epoch + 1

        if (epoch > 0) and (epoch % n == 0) and (self.n_times_trained_p > 0):
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


    def __get_learning_rate(self, optimizer):
        """
        Function for retrieving the learning rate of the optimizer

        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]


    def __resource_cleanup(self):
        """
        Function for cleaning up the resources used by the training session.
        If the device used for training is a GPU, the CUDA cache is cleared.

        """

        if self.device.type == "cuda":
            torch.cuda.empty_cache()


    def __set_seed(self):
        """
        Function for setting the seed for the random number generators.

        """
        try:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        except Exception as e:
            print(e)


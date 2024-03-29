"""
This module contains the Training class.
This class is used to train the generator and predictor models or only the predictor model on the dataset.

When training the generator and predictor models, the training is done using the adversarial training approach.

"""

import numpy as np

import os
import random
import time
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import datetime

from configs.paths import TRAINED_MODELS_DIR
from configs.constants import *

from src.common.folder_manager import FolderManager
from src.common.device_manager import DeviceManager
from src.common.training_base  import TrainingBase
from src.common.model_manager  import ModelManager

from src.common.utils.losses import WeightedMSELoss

from src.common.utils.helpers import get_elapsed_time_str

from src.gol_adv_sys.utils.helpers import get_data_tensor, generate_new_batches, \
                                          test_models, save_progress_plot, get_config_from_batch


class TrainingAdversarial(TrainingBase):
    """
    Class designed to handle the training of the generator and predictor models in an adversarial training approach.
    It can also train only the predictor model on the dataset.

    Attributes:
        seed (int): The seed used for random number generation.
        folders (FolderManager): An instance of FolderManager to manage the creation of folders for the training session.
        device_manager (DeviceManager): An instance of DeviceManager to manage device selection.
        simulation_topology (str): The topology of the simulation grid.
        init_config_initial_type (str): The type of initial configuration to use.
        config_type_pred_target (str): The type of configuration to predict.
        current_epoch (int): The current epoch of the training session.
        step_times_secs (list): A list of lists containing the time in seconds for each step in each epoch.
        losses (dict): A dictionary containing the losses of the generator and predictor models.
        lr_each_epoch (dict): A dictionary containing the learning rate of the generator and predictor models for each epoch.
        n_times_trained_p (int): The number of times the predictor model has been trained.
        n_times_trained_g (int): The number of times the generator model has been trained.
        generator (ModelManager): An instance of ModelManager for the generator model.
        predictor (ModelManager): An instance of ModelManager for the predictor model.
        fixed_noise (torch.Tensor): The fixed noise used for generating configurations.
        properties_g (dict): A dictionary containing properties of the generator model.
        data_tensor (torch.Tensor): Auxiliary tensor used for creating the dataloader.
        path_log_file (str): The path to the log file for the training session.
        path_p (str): The path to the saved predictor model.
        path_g (str): The path to the saved generator model.

    """

    def __init__(self, model_p=None, model_g=None) -> None:
        self.__date = datetime.datetime.now()

        self.__initialize_seed()

        self.folders        = FolderManager(self.__date)
        self.device_manager = DeviceManager()

        self.simulation_topology      = TOPOLOGY_TOROIDAL
        self.init_config_initial_type = INIT_CONFIG_INTIAL_THRESHOLD
        self.config_type_pred_target  = CONFIG_METRIC_MEDIUM

        self.n_times_trained_p = 0
        self.n_times_trained_g = 0
        self.current_epoch     = 0
        self.step_times_secs   = []

        self.losses        = {GENERATOR: [], PREDICTOR: []}
        self.lr_each_epoch = {PREDICTOR: [], GENERATOR: []}


        criterion = WeightedMSELoss()

        self.generator = ModelManager(model=model_g,
                                      optimizer=optim.AdamW(model_g.parameters(),
                                                    lr=G_ADAMW_LR,
                                                    betas=(G_ADAMW_B1, G_ADAMW_B2),
                                                    eps=G_ADAMW_EPS,
                                                    weight_decay=G_ADAMW_WEIGHT_DECAY),
                                      criterion=lambda x, y: -criterion(x, y),
                                      device_manager=self.device_manager)

        self.predictor = ModelManager(model=model_p,
                                      optimizer=optim.SGD(model_p.parameters(),
                                                    lr=P_SGD_LR,
                                                    momentum=P_SGD_MOMENTUM,
                                                    weight_decay=P_SGD_WEIGHT_DECAY),
                                      criterion=criterion,
                                      device_manager=self.device_manager)


        self.fixed_noise  = torch.randn(BATCH_SIZE, N_Z, 1, 1, device=self.device_manager.default_device)

        self.properties_g = {"enabled": True, "can_train": False}

        self.data_tensor  = None

        self.path_log_file = self.__init_log_file()
        self.path_p        = None
        self.path_g        = None


    def run(self) -> None:
        """
        Function used for running the training session.

        """
        torch.autograd.set_detect_anomaly(True)

        logging.info(f"Training started at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        self._fit()

        logging.info(f"Training ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


    def _fit(self) -> None:
        """
        Adversarial training loop.

        """

        for epoch in range(NUM_EPOCHS):

            self.step_times_secs.append([])
            self.current_epoch = epoch

            self.__get_train_dataloader()

            with open(self.path_log_file, "a") as log:

                log.write(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}\n")
                log.write(f"Number of generated configurations in the dataset: {len(self.train_dataloader)*BATCH_SIZE}\n\n")
                log.flush()

            for step in range(NUM_TRAINING_STEPS):

                step_start_time = time.time()

                self.__train_predictor()

                if self.properties_g["enabled"] and self.properties_g["can_train"]:
                    self.__train_generator()

                step_end_time = time.time()
                self.step_times_secs[self.current_epoch].append(step_end_time - step_start_time)

                self.__log_training_step(step)

            self.__log_training_epoch()

            # Update properties for G
            self.__can_g_train()

            # Test and save models
            data = self.__test_models()
            self.__save_progress_plot(data)
            self.__save_models()
            self.__resource_cleanup()


    def __log_training_step(self, step) -> None:
        """
        Log the progress of the training session inside each epoch.

        Args:
            step (int): The current step in the epoch.

        """
        str_step_time = f"{get_elapsed_time_str(self.step_times_secs[self.current_epoch][step])}"
        str_step      = f"{step+1}/{NUM_TRAINING_STEPS}"
        str_err_p     = f"{self.losses[PREDICTOR][-1]}"
        str_err_g     = f"{self.losses[GENERATOR][-1]}" if len(self.losses[GENERATOR]) > 0 else "N/A"


        with open(self.path_log_file, "a") as log:

            log.write(f"{str_step_time} | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}\n")
            log.flush()


    def __log_training_epoch(self) -> None:
        """
        Log the progress of the training session at the end of each epoch.

        """

        with open(self.path_log_file, "a") as log:
            log.write(f"\nElapsed time: {get_elapsed_time_str(self.step_times_secs[self.current_epoch])}\n")

            if self.n_times_trained_p > 0:
                log.write(f"Average loss P: {self.__get_loss_avg_p_last_epoch()}\n")

            if self.n_times_trained_g > 0:
                log.write(f"Average loss G: {self.__get_loss_avg_g_last_epoch()}\n")

            if self.current_epoch + 1 == NUM_EPOCHS:
                 log.write(f"\n\nTraining ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                 log.write(f"Number of times P was trained: {self.n_times_trained_p}\n")
                 log.write(f"Number of times G was trained: {self.n_times_trained_g}\n")

            log.flush()


    def __init_log_file(self) -> str:
        """
        Create a log file for the training session and write the initial specifications.

        Returns:
            path (str): The path to the log file.

        """

        path = self.folders.logs_folder / FILE_NAME_TRAINING_PROGRESS

        seed_info = ("Random seed" if self.__seed_type["random"] else
                     "Fixed seed" if self.__seed_type["fixed"] else
                     "Unknown seed")

        balanced_gpu_info = (f"Balanced GPU indices: {self.device_manager.balanced_gpu_indices}\n"
                             if self.device_manager.balanced_gpu_indices else "")

        topology_info = ("Topology: toroidal" if self.simulation_topology == TOPOLOGY_TOROIDAL else
                         "Topology: flat" if self.simulation_topology == TOPOLOGY_FLAT else
                         "Topology: unknown")

        init_config_initial_type = self.init_config_initial_type
        init_config_info = ""
        if init_config_initial_type == INIT_CONFIG_INTIAL_THRESHOLD:
            init_config_info = (f"Initial configuration type: threshold\n"
                                f"Threshold for the value of the cells: {THRESHOLD_CELL_VALUE}\n")
        elif init_config_initial_type == INIT_CONFIG_INITAL_N_CELLS:
            init_config_info = (f"Initial configuration type: n_living_cells\n"
                                f"Number of living cells in initial grid: {N_LIVING_CELLS_VALUE}\n")
        else:
            init_config_info = "Initial configuration type: unknown\n"

        generator_info = ""
        if self.properties_g["enabled"]:
            generator_info += (f"Latent space size: {N_Z}\n"
                               f"Optimizer G: {self.generator.optimizer.__class__.__name__}\n"
                               f"Criterion G: {self.generator.criterion.__class__.__name__}\n")

        log_contents = (
            f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"{seed_info}: {self.__seed}\n"
            f"Default device: {self.device_manager.default_device}\n"
            f"{balanced_gpu_info}\n"
            f"Training specs:\n"
            f"Batch size: {BATCH_SIZE}\n"
            f"Epochs: {NUM_EPOCHS}\n"
            f"Number of training steps in each epoch: {NUM_TRAINING_STEPS}\n"
            f"Number of batches generated in each epoch: {N_BATCHES} ({N_CONFIGS} configs)\n"
            f"Max number of generated batches in dataset: {N_MAX_BATCHES} ({N_MAX_CONFIGS} configs)\n"
            f"\nSimulation specs:\n"
            f"Grid size: {GRID_SIZE}\n"
            f"Simulation steps: {N_SIM_STEPS}\n"
            f"{topology_info}\n"
            f"{init_config_info}"
            f"\nPredicting config type: {self.config_type_pred_target}\n"
            f"\nModel specs:\n"
            f"Optimizer P: {self.predictor.optimizer.__class__.__name__}\n"
            f"Criterion P: {self.generator.criterion.__class__.__name__}\n"
            f"{generator_info}"
            f"\nTraining progress:\n\n"
        )

        with open(path, "w") as log_file:
            log_file.write(log_contents.strip())
            log_file.flush()

        return path


    def __get_train_dataloader(self) -> DataLoader:
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

        data = get_data_tensor(self.data_tensor, self.generator.model,
                               self.simulation_topology, self.init_config_initial_type, self.device_manager.default_device)

        self.data_tensor = data

        # Create the dataloader from the tensor
        self.train_dataloader = DataLoader(self.data_tensor, batch_size=BATCH_SIZE, shuffle=True)

        return self.train_dataloader


    def __get_new_batches(self, n_batches) -> torch.Tensor:
        """
        Generate new configurations using the generator model.

        Args:
            n_batches (int): The number of batches to generate.

        Returns:
            new_configs (list): A list of dictionaries containing information about the generated configurations.

        """

        return  generate_new_batches(self.generator.model, n_batches, self.simulation_topology,
                                     self.init_config_initial_type, self.device_manager.default_device)


    def __get_one_new_batch(self) -> torch.Tensor:
        """
        Generate one new batch of configurations using the generator model.

        Returns:
            new_config (dict): A dictionary containing information about the generated configurations.

        """

        return self.__get_new_batches(1)


    def __get_config_type(self, batch, type) -> torch.Tensor:
        """
        Function to get a batch of the specified configuration type from the batch.

        Returns:
            config (torch.Tensor): The configurations.

        """

        return get_config_from_batch(batch, type, self.device_manager.default_device)


    def __train_predictor(self) -> float:
        """
        Function for training the predictor model.

        Returns:
            loss (float): The loss of the predictor model.

        """

        loss = 0
        self.predictor.model.train()

        for batch in self.train_dataloader:
            self.predictor.optimizer.zero_grad()
            predicted = self.predictor.model(self.__get_config_type(batch, CONFIG_INITIAL))
            errP = self.predictor.criterion(predicted , self.__get_config_type(batch, self.config_type_pred_target))
            errP.backward()
            self.predictor.optimizer.step()
            loss += errP.item()

        self.n_times_trained_p += 1

        loss /= len(self.train_dataloader)
        self.losses[PREDICTOR].append(loss)

        return loss


    def __train_generator(self) -> float:
        """
        Function for training the generator model.

        Returns:
            loss (float): The loss of the generator model.

        """

        loss = 0
        n = N_BATCHES

        self.generator.model.train()

        for _ in range(n):
            batch = self.__get_one_new_batch()
            self.generator.optimizer.zero_grad()
            predicted = self.model_p(self.__get_config_type(batch, CONFIG_FINAL))
            errG = self.criterion_g(predicted, self.__get_config_type(batch, self.config_type_pred_target))
            errG.backward()
            self.generator.optimizer.step()
            loss += (-1 * errG.item())

        self.n_times_trained_g += 1

        loss /= n
        self.losses[GENERATOR].append(loss)

        return loss


    def __get_loss_avg_p(self, on_last_n_losses) -> float:
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

        len_losses_p_train = len(self.losses[PREDICTOR])
        if len_losses_p_train <= 0:
            return None

        if on_last_n_losses > len_losses_p_train:
            on_last_n_losses = len_losses_p_train

        avg_loss_p = sum(self.losses[PREDICTOR][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_p


    def __get_loss_avg_g(self, on_last_n_losses) -> float:
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

        len_losses_g = len(self.losses["generator"])
        if len_losses_g <= 0:
            return None

        if on_last_n_losses > len_losses_g:
            on_last_n_losses = len_losses_g

        avg_loss_g = sum(self.losses["generator"][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_g


    def __get_loss_avg_p_last_epoch(self) -> float:
        """
        Special case of __get_loss_avg_p() where the average is calculated for the last epoch.

        Returns:
            avg_loss_p_last_epoch (float): The average loss of the predictor model on the last epoch.

        """

        return self.__get_loss_avg_p(NUM_TRAINING_STEPS)


    def __get_loss_avg_g_last_epoch(self) -> float:
        """
        Special case of __get_loss_avg_g() where the average is calculated for the last epoch.

        Returns:
            avg_loss_g_last_epoch (float): The average loss of the generator model on the last epoch.

        """

        return self.__get_loss_avg_g(NUM_TRAINING_STEPS)


    def __test_models(self) -> dict:
        """
        Function for testing the models.
        The models are tested on the fixed noise.

        Returns:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated metrics and predicted metrics.

        """

        return test_models(self.generator.model, self.predictor.model, self.simulation_topology,
                           self.init_config_initial_type, self.fixed_noise, self.config_type_pred_target,
                           self.device_manager.default_device)


    def __save_progress_plot(self, data) -> None:
        """
        Function for saving the progress plot.
        It save the plot that shows the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

        Args:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated metrics and predicted metrics.

        """

        save_progress_plot(data, self.current_epoch, self.folders.results_folder)


    def __save_models(self) -> None:
        """
        Function for saving the models.

        It saves the generator and predictor models to the models folder every n times they are trained.

        """

        n = 10
        epoch = self.current_epoch + 1

        if (epoch > 0) and (epoch % n == 0) and (self.n_times_trained_p > 0):
            self.path_p = self.folders.models_folder / f"predictor_{epoch}.pth.tar"
            self.predictor.save(self.path_p)


        if (epoch > 0) and (epoch % n == 0) and (self.n_times_trained_g > 0):
            self.path_g = self.folders.models_folder / f"generator_{epoch}.pth.tar"
            self.generator.save(self.path_g)


    def __can_g_train(self) -> bool:
        """
        Function that returns if the generator model can be trained.
        The generator model can be trained if the average loss of the predictor model is less than a certain threshold.

        Returns:
            can_train (bool): True if the generator model can be trained, False otherwise.

        """

        if not self.properties_g["enabled"]:
            return False

        if self.properties_g["enabled"] and self.properties_g["can_train"]:
            return True

        if self.properties_g["enabled"] and not self.properties_g["can_train"]:
            self.properties_g["can_train"] = self.__get_loss_avg_p_last_epoch() < THRESHOLD_AVG_LOSS_P

            if self.properties_g["can_train"]:
                # +1 because the current epoch is 0-based, +2 because the generator can be trained from the next epoch
                logging.info(f"Generator can start training from next epoch. The next epoch is number {self.current_epoch + 2}")

        return self.properties_g["can_train"]


    def __resource_cleanup(self) -> None:
        """
        Function for cleaning up the resources used by the training session.
        If the device used for training is a GPU, the CUDA cache is cleared.

        """

        if self.device.type == "cuda":
            torch.cuda.empty_cache()


    def __initialize_seed(self) -> None:
        """
        Initialize the seed for the random number generators.

        """

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.__seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]

        try:
            random.seed(self.__seed)
            torch.manual_seed(self.__seed)
            np.random.seed(self.__seed)
            torch.cuda.manual_seed(self.__seed)
        except Exception as e:
            logging.error(f"Error initializing the seed: {e}")


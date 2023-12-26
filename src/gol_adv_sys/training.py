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

import datetime

from .utils import constants as constants

from .model_p import Predictor, Predictor_18
from .model_g import Generator

from .utils.folders import training_folders
from .utils.helper_functions import test_models, save_progress_plot, get_epoch_elapsed_time_str, generate_new_batches, get_dataloader


class Training():
    def __init__(self):
        self.__date = datetime.datetime.now()
        self.folders = training_folders(self.__date)

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]
        self.__set_seed()

        self.n_devices = torch.cuda.device_count()
        self.device = self.__choose_device()
        self.balanced_gpu_indices = self.__get_balanced_gpu_indices()

        self.simulation_topology = constants.TOPOLOGY_TYPE["toroidal"]
        self.init_conf_type = constants.INIT_CONF_TYPE["n_living_cells"]

        self.dataloader = []
        self.current_epoch = 0
        self.step_times_secs = []
        self.properties_g= {"enabled": True, "can_train": False}

        self.results = {"losses_g": [],
                        "losses_p": []}

        self.n_times_trained_p = 0
        self.n_times_trained_g = 0

        self.criterion_p = nn.MSELoss()
        self.criterion_g = lambda x, y: -self.criterion_p(x, y)

        self.model_p = Predictor().to(self.device)
        self.model_g = Generator().to(self.device)

        self.optimizer_p = optim.AdamW(self.model_p.parameters(),
                                       lr=constants.p_adamw_lr,
                                       betas=(constants.p_adamw_b1, constants.p_adamw_b2),
                                       eps=constants.p_adamw_eps,
                                       weight_decay=constants.p_adamw_wd)

        self.optimizer_g = optim.Adam(self.model_g.parameters(),
                                      lr=constants.g_adam_lr,
                                      betas=(constants.g_adam_b1, constants.g_adam_b2),
                                      eps=constants.g_adam_eps)

        self.fixed_noise = torch.randn(constants.bs, constants.nz, 1, 1, device=self.device)

        self.path_log_file = None
        self.path_p        = None
        self.path_g        = None


    """
    Function to run the training.

    """
    def run(self):
        torch.autograd.set_detect_anomaly(True)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) # Needed for reproducible results

        self.__fit()


    """
    Adversarial training loop.

    """
    def __fit(self):

        if len(self.balanced_gpu_indices) > 0:
            self.model_p = nn.DataParallel(self.model_p, device_ids=self.balanced_gpu_indices)
            self.model_g = nn.DataParallel(self.model_g, device_ids=self.balanced_gpu_indices)

        if self.path_log_file is None:
            self.path_log_file = self.__init_log_file()

        with open(self.path_log_file, "a") as log:
            for epoch in range(constants.num_epochs):

                self.step_times_secs.append([])
                self.current_epoch = epoch

                self.__get_dataloader()

                log.write(f"\n\nEpoch: {epoch+1}/{constants.num_epochs}\n")
                log.write(f"Number of generated configurations in data set: {len(self.dataloader)*constants.bs}\n\n")
                log.flush()

                for step in range(constants.num_training_steps):

                    step_start_time = time.time()

                    self.__train_predictor()

                    if self.properties_g["enabled"]:
                        self.__train_generator()

                    step_end_time = time.time()
                    self.step_times_secs[self.current_epoch].append(step_end_time - step_start_time)

                    self.__log_training_step(step)


                log.write(f"\nElapsed time: {get_epoch_elapsed_time_str(self.step_times_secs[epoch])}\n")
                log.write(f"Average loss P: {self.__get_loss_avg_p(constants.num_training_steps)}\n")
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
    Log the progress of the training session inside each epoch.

    """
    def __log_training_step(self, step):
        with open(self.path_log_file, "a") as log:

            str_step      = f"{step+1}/{constants.num_training_steps}"
            str_step_time = f"{self.step_times_secs[self.current_epoch][step]:.2f} s"
            str_err_p     = f"{self.results['losses_p'][-1]}"
            str_err_g     = f"{self.results['losses_g'][-1]}" if len(self.results['losses_g']) > 0 else "N/A"

            log.write(f"{str_step_time} | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}\n")
            log.flush()


    """
    Create a log file for the training session.
    When creating the log file, the training specifications are also written to the file.

    Returns:
        path (str): The path to the log file.

    """
    def __init_log_file(self):

        path = os.path.join(self.folders.logs_path, "asgntc.txt")

        with open(path, "w") as log_file:
            log_file.write(f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n\n")

            if self.__seed_type["random"]:
                log_file.write(f"Random seed: {self.seed}\n")
            elif self.__seed_type["fixed"]:
                log_file.write(f"Fixed seed: {self.seed}\n")
            else:
                log_file.write(f"Unknown seed\n")

            log_file.write(f"Default device: {self.device}\n")
            if len(self.balanced_gpu_indices) > 0:
                log_file.write(f"Balanced GPU indices: {self.balanced_gpu_indices}\n")

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

            if self.init_conf_type == constants.INIT_CONF_TYPE["threshold"]:
                log_file.write(f"Initial configuration type: threshold\n")
                log_file.write(f"Threshold for the value of the cells: {constants.threshold_cell_value}\n")
            elif self.init_conf_type == constants.INIT_CONF_TYPE["n_living_cells"]:
                log_file.write(f"Initial configuration type: n_living_cells\n")
                log_file.write(f"Number of living cells in initial grid: {constants.n_living_cells}\n")
            else:
                log_file.write(f"Initial configuration type: unknown\n")


            log_file.write(f"\nModels specs:\n")
            log_file.write(f"Latent space size: {constants.nz}\n")
            log_file.write(f"Optimizer P: {self.optimizer_p.__class__.__name__}\n")
            log_file.write(f"Optimizer G: {self.optimizer_g.__class__.__name__}\n")
            log_file.write(f"Criterion P: {self.criterion_p.__class__.__name__}\n")
            log_file.write(f"Criterion G: -{self.criterion_p.__class__.__name__}\n")
            log_file.write(f"Average loss of P before training G: {constants.threshold_avg_loss_p}\n\n")
            log_file.flush()

        return path


    """
    Function for choosing the device to use for training.
    If a GPU is available, the GPU with the most free memory is selected.

    Returns:
        device (torch.device): The device to use for training.
    """
    def __choose_device(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        device_specs = []

        for i in range(self.n_devices):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_free = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
            device_specs.append((i, gpu_properties.total_memory, memory_free))

        selected_device = max(device_specs, key=lambda x: x[2])[0]

        return torch.device(f"cuda:{selected_device}")


    """
    Get GPU indices that are balanced compared to the default device that has been chosen.

    Args:
        threshold (float): The minimum ratio of memory and cores compared to the default device to be considered balanced.

    Returns:
        list: A list of GPU indices that are balanced.
    """
    def __get_balanced_gpu_indices(self, threshold=0.75):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return []

        id_default_device = torch.cuda.current_device()
        default_gpu_properties = torch.cuda.get_device_properties(id_default_device)
        balanced_gpus = []

        for i in range(self.n_devices):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_ratio = gpu_properties.total_memory / default_gpu_properties.total_memory
            cores_ratio = gpu_properties.multi_processor_count / default_gpu_properties.multi_processor_count

            if memory_ratio >= threshold and cores_ratio >= threshold and i != id_default_device:
                balanced_gpus.append(i)

        return balanced_gpus


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
    def __get_dataloader(self):
        self.dataloader = get_dataloader(self.dataloader, self.model_g, self.simulation_topology, self.init_conf_type, self.device)

        return self.dataloader

    """
    Generate new configurations using the generator model.

    Args:
        n_batches (int): The number of batches to generate.

    Returns:
        new_configs (list): A list of dictionaries containing information about the generated configurations.
    """
    def __get_new_batches(self, n_batches):
        return  generate_new_batches(self.model_g, n_batches, self.simulation_topology, self.init_conf_type, self.device)


    """
    Generate one new batch of configurations using the generator model.

    Returns:
        new_config (dict): A dictionary containing information about the generated configurations.
    """
    def __get_one_new_batch(self):
        return self.__get_new_batches(1)

    """
    Function for training the predictor model.

    Returns:
        loss (float): The loss of the predictor model.
    """
    def __train_predictor(self):

        loss = 0
        self.model_p.train()

        for batch in self.dataloader:
            if torch.cuda.is_available():
                batch["initial"] = batch["initial"].to("cuda")
                batch["simulated"]["metric"] = batch["simulated"]["metric"].to("cuda")

            self.optimizer_p.zero_grad()
            predicted_metric = self.model_p(batch["initial"])
            errP = self.criterion_p(predicted_metric, batch["simulated"]["metric"])
            errP.backward()
            self.optimizer_p.step()
            loss += errP.item()

        self.n_times_trained_p += 1

        loss /= len(self.dataloader)
        self.results["losses_p"].append(loss)

        return loss


    """
    Function for training the generator model.

    Returns:
        loss (float): The loss of the generator model.
    """
    def __train_generator(self):

        if self.properties_g["can_train"]:

            loss = 0
            for _ in range(constants.n_batches_g_training):
                self.model_g.train()

                batch = self.__get_one_new_batch()
                if torch.cuda.is_available():
                    batch[0]["initial"] = batch[0]["initial"].to("cuda")
                    batch[0]["simulated"]["metric"] = batch[0]["simulated"]["metric"].to("cuda")

                self.optimizer_g.zero_grad()
                predicted_metric = self.model_p(batch[0]["initial"])
                errG = self.criterion_g(predicted_metric, batch[0]["simulated"]["metric"])
                errG.backward()
                self.optimizer_g.step()
                loss += errG.item()

            self.n_times_trained_g += 1

            loss /= constants.n_batches_g_training
            self.results["losses_g"].append(loss)

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

        len_losses_p = len(self.results["losses_p"])
        if len_losses_p <= 0:
            return None

        if on_last_n_losses > len_losses_p:
            on_last_n_losses = len_losses_p

        avg_loss_p = sum(self.results["losses_p"][-on_last_n_losses:])/on_last_n_losses

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

        len_losses_g = len(self.results["losses_g"])
        if len_losses_g <= 0:
            return None

        if on_last_n_losses > len_losses_g:
            on_last_n_losses = len_losses_g

        avg_loss_g = sum(self.results["losses_g"][-on_last_n_losses:])/on_last_n_losses

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
        return test_models(self.model_g, self.model_p, self.simulation_topology, self.init_conf_type, self.fixed_noise, self.device)


    """
    Function for saving the progress plot.
    It save the plot that shows the generated configurations, initial configurations, simulated configurations,
    simulated metrics and predicted metrics.

    Args:
        data (dict): Contains the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.
    """
    def __save_progress_plot(self, data):
        save_progress_plot(data, self.current_epoch, self.folders.results_path)


    """
    Function for saving the models.
    It saves the generator and predictor models.

    """
    def __save_models(self):

        if self.path_g is None and self.n_times_trained_g > 0:
            self.path_g = os.path.join(self.folders.models_path, "generator.pth.tar")

        if self.path_p is None and self.n_times_trained_p > 0:
            self.path_p = os.path.join(self.folders.models_path, "predictor.pth.tar")


        if self.path_g is not None:
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


        if self.path_p is not None:
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

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

from .train_models import fit

from .utils.folders import training_folders
from .utils.helper_functions import test_models, save_progress_plot, get_epoch_elapsed_time_str, generate_new_configs, get_dataloader


class Training():
    def __init__(self):
        self.__date = datetime.datetime.now()
        self.folders = training_folders(self.__date)

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]

        self.device = self.__choose_device()

        self.fixed_noise = torch.randn(constants.bs, constants.nz, 1, 1, device=self.device)

        self.dataloader = []
        self.n_new_configs = []
        self.current_epoch = 0
        self.step_times_secs = []
        self.properties_g= {"enabled": True, "can_train": False}

        self.results = {"losses_g": [],
                        "losses_p": []}

        self.n_times_trained_p = self.__get__n_times_trained_p()
        self.n_times_trained_g = self.__get__n_times_trained_g()

        self.criterion_p = nn.MSELoss()
        self.criterion_g = lambda x, y: -1 * nn.MSELoss()(x, y)

        self.model_g = Generator().to(self.device)
        self.model_p = Predictor_18().to(self.device)

        self.optimizer_g = optim.Adam(self.model_g.parameters(),
                                      lr=constants.g_adam_lr,
                                      betas=(constants.g_adam_b1, constants.g_adam_b2),
                                      eps=constants.g_adam_eps)

        self.optimizer_p = optim.AdamW(self.model_p.parameters(),
                                        lr=constants.p_adamw_lr,
                                        betas=(constants.p_adamw_b1, constants.p_adamw_b2),
                                        eps=constants.p_adamw_eps,
                                        weight_decay=constants.p_adamw_wd)


        self.log_file = self.__create_log_file()
        self.path_p = None
        self.path_g = None


    # TODO: get_training_specs()
    """
    Get the specifications of the training session.

    Returns:
        training_specs (dict): The specifications of the training session.

    """
    def get_training_specs(self):
        return {
            "seed": self.seed,
            "device": self.device,
            "batch size": constants.bs,
            "epochs": constants.num_epochs,
            "grid size": constants.grid_size,
            "nz": constants.nz,
        }


    """
    Function to run the training.

    Returns:
        result (dict): The results of the training session.

    """
    def run(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.use_deterministic_algorithms(True) # Needed for reproducible results
        torch.autograd.set_detect_anomaly(True)

        # Fit
        # G_losses, P_losses = fit(self.model_g, self.model_p, self.optimizer_g, self.optimizer_p,
        #                          self.criterion_g, self.criterion_p, self.fixed_noise, self.folders,
        #                          self.log_file, self.device)

        # FIT
        self.__fit()


    # PRIVATE METHODS


    def __fit(self):
        with open(self.log_file, "a") as log:
            for epoch in range(constants.num_epochs):

                self.step_times_secs.append([])
                self.current_epoch = epoch

                self.__get_dataloader()

                log.write(f"\n\nEpoch: {epoch+1}/{constants.num_epochs}\n")
                log.write(f"Number of generated configurations in data set: {len(self.dataloader)*constants.bs}\n")
                log.flush()

                for step in range(constants.num_training_steps):

                    step_start_time = time.time()

                    self.__train_predictor()

                    if self.properties_g["enabled"]:
                        self.__train_generator()

                    step_end_time = time.time()
                    self.step_times_secs[self.current_epoch].append(step_end_time - step_start_time)

                    self.__log_training_step(step)


                log.write(f"Elapsed time: {get_epoch_elapsed_time_str(self.step_times_secs[epoch])}\n")
                log.write(f"Avg loss P: {self.__get_loss_avg_p(constants.num_training_steps)}\n")
                log.flush()

                # Update properties for G
                self.__can_g_train()

                data = self.__test_models()
                self.__save_progress_plot(data)
                self.__save_models()

                # Clear CUDA cache
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()


    """
    Create a log file for the training session.
    When creating the log file, the training specifications are also written to the file.

    Returns:
        log_file_path (str): The path to the log file.

    """
    def __create_log_file(self):
        log_file_path = os.path.join(self.folders.logs_path, "asgntc.txt")

        log_file = open(log_file_path, "w")
        log_file.write(f"Date: {self.__date}\n")
        log_file.write(f"Seed: {self.seed}\n")
        log_file.write(f"Device: {self.device}\n")
        log_file.write(f"Epochs: {constants.num_epochs}\n")
        log_file.write(f"Batch size: {constants.bs}\n")
        log_file.write(f"Grid size: {constants.grid_size}x{constants.grid_size}\n")
        log_file.write(f"Simulation steps: {constants.n_simulation_steps}\n")
        log_file.write(f"Threshold cell value: {constants.threshold_cell_value}\n")
        log_file.write("Models specs:\n")
        log_file.write(f"nc: {constants.nc}\n")
        log_file.write(f"ngf: {constants.ngf}\n")
        log_file.write(f"ndf: {constants.ndf}\n")
        log_file.write(f"nz: {constants.nz}\n")
        log_file.write(f"Optimizer P: AdamW\n")
        log_file.write(f"Optimizer G: Adam\n")
        log_file.write(f"Criterion P: MSELoss\n")
        log_file.write(f"Criterion G: -1 * MSELoss\n")
        log_file.write(f"Threshold avg loss P before training G: {constants.threshold_avg_loss_p}\n")
        log_file.flush()

        log_file.close()

        return log_file_path


    def __choose_device(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        num_gpus = torch.cuda.device_count()
        device_specs = []

        for i in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_free = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
            device_specs.append((i, gpu_properties.total_memory, memory_free))

        # Select a GPU based on the amount of free memory
        selected_device = max(device_specs, key=lambda x: x[2])[0]

        return torch.device(f"cuda:{selected_device}")


    def __log_training_step(self, step):
        with open(self.log_file, "a") as log:

            str_step = f"{step+1}/{constants.num_training_steps}"
            str_step_time = f"{self.step_times_secs[self.current_epoch][step]:.2f} s"
            str_err_p = f"{self.results['losses_p'][-1]}"
            str_err_g = f"{self.results['losses_g'][-1]}" if len(self.results['losses_g']) > 0 else "N/A"

            log.write(f"{str_step_time} | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}\n")
            log.flush()


    def __get_dataloader(self):
        self.dataloader = get_dataloader(self.dataloader, self.model_g, self.device)

        return self.dataloader


    def __get_n_new_configs(self):
        self.n_new_configs = generate_new_configs(self.model_g, constants.n_configs, self.device)

        return self.n_new_configs


    def __train_predictor(self):

        loss = 0
        self.model_p.train()
        for config in self.dataloader:
            self.optimizer_p.zero_grad()
            predicted_metric = self.model_p(config["generated"].detach())
            errP = self.criterion_p(predicted_metric, config["simulated"]["metric"])
            errP.backward()
            self.optimizer_p.step()
            loss += errP.item()

        self.n_times_trained_p += 1

        loss /= len(self.dataloader)
        self.results["losses_p"].append(loss)

        return loss


    def __train_generator(self):

        if self.properties_g["can_train"]:

            loss = 0
            self.model_g.train()
            self.__get_n_new_configs()
            for config in self.n_new_configs:
                self.optimizer_g.zero_grad()
                predicted_metric = self.model_p(config["generated"])
                errG = self.criterion_g(predicted_metric, config["simulated"]["metric"])
                errG.backward()
                self.optimizer_g.step()
                loss += errG.item()

            self.n_times_trained_g += 1

            loss /= len(self.n_new_configs)
            self.results["losses_g"].append(loss)

            return loss

        else:
            return None


    def __get_loss_avg_p(self, on_last_n_losses):

        len_losses_p = len(self.results["losses_p"])
        if len_losses_p <= 0:
            return None

        if on_last_n_losses > len_losses_p:
            on_last_n_losses = len_losses_p

        avg_loss_p = sum(self.results["losses_p"][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_p


    def __get_loss_avg_g(self, on_last_n_losses):

        len_losses_g = len(self.results["losses_g"])
        if len_losses_g <= 0:
            return None

        if on_last_n_losses > len_losses_g:
            on_last_n_losses = len_losses_g

        avg_loss_g = sum(self.results["losses_g"][-on_last_n_losses:])/on_last_n_losses

        return avg_loss_g


    def __test_models(self):
        return test_models(self.model_g, self.model_p, self.fixed_noise, self.device)


    def __save_progress_plot(self, data):
        save_progress_plot(data, self.current_epoch, self.folders.results_path)


    def __save_models(self):

        if self.path_g is None:
            self.path_g = os.path.join(self.folders.models_path, "generator.pth.tar")

        if self.path_p is None:
            self.path_p = os.path.join(self.folders.models_path, "predictor.pth.tar")

        torch.save({
                    "state_dict": self.model_g.state_dict(),
                    "optimizer": self.optimizer_g.state_dict(),
                   }, self.path_g)

        torch.save({
                    "state_dict": self.model_p.state_dict(),
                    "optimizer": self.optimizer_p.state_dict(),
                   }, self.path_p)


    def __get__n_times_trained_g(self):
        return len(self.results["losses_g"])


    def __get__n_times_trained_p(self):
        return len(self.results["losses_p"])


    def __can_g_train(self):

        if not self.properties_g["enabled"]:
            return False

        if self.properties_g["enabled"] and self.properties_g["can_train"]:
            return True

        if self.properties_g["enabled"] and not self.properties_g["can_train"]:
            self.properties_g["can_train"] = self.__get_loss_avg_p(self.n_times_trained_p) < constants.threshold_avg_loss_p

        return self.properties_g["can_train"]


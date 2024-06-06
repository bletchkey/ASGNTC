"""
This module contains the Training class.
This class is used to train the generator and predictor models or only the predictor model on the dataset.

When training the generator and predictor models, the training is done using the adversarial training approach.

"""

import numpy as np

import os
import gc
import random
import time
from pathlib import Path
from typing  import Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


import datetime

from configs.paths import TRAINED_MODELS_DIR
from configs.constants import *

from src.common.folder_manager import FolderManager
from src.common.device_manager import DeviceManager
from src.common.training_base  import TrainingBase
from src.common.model_manager  import ModelManager

from src.common.utils.losses   import WeightedMSELoss, AdversarialGoLLoss

from src.common.utils.helpers  import get_elapsed_time_str

from src.gol_adv_sys.utils.helpers import generate_new_batches, \
                                          generate_new_training_batches, \
                                          test_models, \
                                          save_progress_plot, get_config_from_batch,\
                                          get_config_from_training_batch, \
                                          get_dirichlet_input_noise


class TrainingAdversarial(TrainingBase):
    """
    Class designed to handle the training of the generator and predictor models in an adversarial training approach.
    It can also train only the predictor model on the dataset.

    Attributes:
        seed (int): The seed used for random number generation.
        folders (FolderManager): An instance of FolderManager to manage the creation of folders for the training session.
        device_manager (DeviceManager): An instance of DeviceManager to manage device selection.
        simulation_topology (str): The topology of the simulation grid.
        config_type_pred_target (str): The type of configuration to predict.
        init_config_initial_type (str): The type of initilization for the initial configuration to use.
        current_iteration (int): The current iteration of the training session.
        step_times_secs (list): A list of lists containing the time in seconds for each step in each iteration.
        complexity_stable_targets (list): A list containing the avg complexity of stable targets of the new generated configurations.
        losses (dict): A dictionary containing the losses of the generator and predictor models.
        lr_each_iteration (dict): A dictionary containing the learning rate of the generator and predictor models for each iteration.
        n_times_trained_p (int): The number of times the predictor model has been trained.
        n_times_trained_g (int): The number of times the generator model has been trained.
        generator (ModelManager): An instance of ModelManager for the generator model.
        predictor (ModelManager): An instance of ModelManager for the predictor model.
        data_loader (torch.utils.data.DataLoader): The dataloader for the training session.
        fixed_input_noise (torch.Tensor): The fixed noise given as input to the generator model.
        properties_g (dict): A dictionary containing properties of the generator model.
        path_log_file (str): The path to the log file for the training session.

    """

    def __init__(self, model_p=None, model_g=None) -> None:
        self.__date = datetime.datetime.now()

        self.__initialize_seed()

        self.folders        = FolderManager(TRAINING_TYPE_ADVERSARIAL, self.__date)
        self.device_manager = DeviceManager()

        self.simulation_topology      = TOPOLOGY_TOROIDAL
        self.config_type_pred_target  = CONFIG_TARGET_EASY
        self.init_config_initial_type = INIT_CONFIG_INITIAL_SIGN
        self.fixed_input_noise        = get_dirichlet_input_noise(ADV_BATCH_SIZE,
                                                                  self.device_manager.default_device)

        self.n_times_trained_p = 0
        self.n_times_trained_g = 0
        self.current_iteration = 0
        self.step_times_secs   = []

        self.complexity_stable_targets= []

        self.losses            = {GENERATOR: [], PREDICTOR: []}
        self.lr_each_iteration = {PREDICTOR: [], GENERATOR: []}

        self.generator = ModelManager(model=model_g,
                                      optimizer=optim.AdamW(model_g.parameters(),
                                                    lr=G_ADAMW_LR,
                                                    betas=(G_ADAMW_B1, G_ADAMW_B2),
                                                    eps=G_ADAMW_EPS,
                                                    weight_decay=G_ADAMW_WEIGHT_DECAY),
                                      criterion = AdversarialGoLLoss(model_type=GENERATOR),
                                      type= GENERATOR,
                                      device_manager=self.device_manager)

        self.predictor = ModelManager(model=model_p,
                                      optimizer=optim.SGD(model_p.parameters(),
                                                    lr=P_SGD_LR,
                                                    momentum=P_SGD_MOMENTUM,
                                                    weight_decay=P_SGD_WEIGHT_DECAY),
                                      criterion = AdversarialGoLLoss(model_type=PREDICTOR),
                                      type=PREDICTOR,
                                      device_manager=self.device_manager)


        self.train_dataloader = None

        self.properties_g  = {"enabled": True, "can_train": True}
        self.path_log_file = self.__init_log_file()


    def run(self) -> None:
        """
        Function used for running the training session.

        """
        torch.autograd.set_detect_anomaly(True)

        logging.info(f"Adversarial training started at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        self._fit()

        logging.info(f"Adversarial training ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


    def _fit(self) -> None:
        """
        Adversarial training loop.

        """

        self.__warmup_predictor()

        for iteration in range(NUM_ITERATIONS):

            self.__get_train_dataloader()

            logging.info(f"Adversarial training iteration {iteration+1}/{NUM_ITERATIONS}")

            self.step_times_secs.append([])
            self.current_iteration = iteration

            with open(self.path_log_file, "a") as log:
                log_content = (
                    f"\n\nIteration: {iteration+1}/{NUM_ITERATIONS}\n"
                    f"Number of generated configurations in the dataset: {len(self.train_dataloader) * ADV_BATCH_SIZE}\n"
                )

                # if len(self.complexity_stable_targets) > 0:
                #     log_content += (
                #         f"Average complexity of the stable targets on the last "
                #         f"{NUM_BATCHES * ADV_BATCH_SIZE} generated configurations: "
                #         f"{100*self.complexity_stable_targets[-1]:.1f}/100\n\n"
                #     )

                log.write(log_content)
                log.flush()

            for step in range(NUM_TRAINING_STEPS):

                step_start_time = time.time()

                self.__train_predictor()

                if self.properties_g["enabled"] and self.properties_g["can_train"]:
                    self.__train_generator()

                step_end_time = time.time()
                self.step_times_secs[self.current_iteration].append(step_end_time - step_start_time)

                self.__log_training_step(step)

            self.__log_training_iteration()

            # Update properties for G
            self.__can_g_train()

            # Test and save models
            data = self.__test_iteration_progress()
            self.__save_progress_plot(data)
            self.__save_models()

            self.device_manager.clear_resources()


    def __get_train_dataloader(self) -> DataLoader:
        """
        Get the dataloader for the current iteration.

        Each iteration, a new dataloader is created by adding n_batches new configurations to the dataloader, for a total of
        n_configs new configurations.
        The maximum number of batches in the dataloader is n_max_batches, that contains n_max_configs configurations.
        The older batches of configurations are removed to make room for the new ones.

        The configurations are generated by the generator model.

        """

        generated, target = self.__get_new_training_batches(NUM_BATCHES)

        if self.train_dataloader == None:
            self.train_dataloader = DataLoader(TensorDataset(generated, target), batch_size=ADV_BATCH_SIZE, shuffle=True)
        else:
            new_dataset           = TensorDataset(generated, target)
            combined_dataset      = ConcatDataset([self.train_dataloader.dataset, new_dataset])
            self.train_dataloader = DataLoader(combined_dataset, batch_size=ADV_BATCH_SIZE, shuffle=True)


    def __warmup_predictor(self) -> None:

        logging.debug(f"Warmup of the predictor model started")

        self.predictor.model.train()

        generated, target = self.__get_new_training_batches(NUM_BATCHES)
        dataset           = TensorDataset(generated, target)
        dataloader_warmup = DataLoader(dataset, batch_size=ADV_BATCH_SIZE, shuffle=True)

        warmup_values = np.linspace(WARMUP_INITIAL_LR, WARMUP_TARGET_LR, len(dataloader_warmup))

        for batch_count, (generated, target) in enumerate(dataloader_warmup, start=1):

            self.predictor.set_learning_rate(warmup_values[batch_count-1])

            logging.debug(f"Warmup phase => Predictor learning rate: {self.predictor.get_learning_rate()}")

            self.predictor.optimizer.zero_grad()

            predicted = self.predictor.model(generated.detach())
            errP      = self.predictor.criterion(predicted, target.detach())

            errP.backward()
            self.predictor.optimizer.step()

        logging.debug(f"Warmup phase => ended")


    def __train_predictor(self, num_epochs: int=5) -> float:
        """
        Function for training the predictor model.

        Args:
            num_epochs (int): Number of times to iterate over the training dataset.

        Returns:
            float: The average loss of the predictor model after all epochs.

        """

        total_loss = 0

        for epoch in range(num_epochs):
            epoch_loss = 0

            logging.debug(f"Training predictor - Epoch {epoch+1}")
            self.predictor.model.train()

            for batch_count, (generated, target) in enumerate(self.train_dataloader, start=1):

                self.predictor.optimizer.zero_grad()

                predicted = self.predictor.model(generated.detach())
                errP = self.predictor.criterion(predicted, target.detach())

                errP.backward()
                self.predictor.optimizer.step()

                epoch_loss += errP.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.losses[PREDICTOR].append(avg_epoch_loss)
            total_loss += epoch_loss

            logging.debug(f"Completed Epoch {epoch+1}: Average Loss {avg_epoch_loss}")

        self.n_times_trained_p += 1

        return total_loss / num_epochs


    def __train_generator(self) -> float:
        """
        Function for training the generator model.

        Returns:
            loss (float): The loss of the generator model.

        """

        loss = 0

        logging.debug(f"Training generator")
        self.generator.model.train()

        for batch_count in range(1, NUM_BATCHES+1):

            self.generator.optimizer.zero_grad()

            generated, target = self.__get_new_training_batches(n_batches=1)

            predicted = self.predictor.model(generated)
            errG      = self.generator.criterion(predicted, target)

            errG.backward()
            self.generator.optimizer.step()

            loss += errG.item()
            running_avg_loss = loss / batch_count

        self.n_times_trained_g += 1

        self.losses[GENERATOR].append(running_avg_loss)

        return loss


    def __get_new_batches(self, n_batches) -> Tuple[torch.Tensor, list]:
        """
        Generate new configurations using the generator model.

        Args:
            n_batches (int): The number of batches to generate.

        Returns:
            new_configs (list): A list of dictionaries containing information about the generated configurations.

        """

        return  generate_new_batches(self.generator.model,
                                     n_batches,
                                     self.simulation_topology,
                                     self.init_config_initial_type,
                                     self.device_manager.default_device)


    def __get_new_training_batches(self, n_batches) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new configurations using the generator model for adversarial training.

        Args:
            n_batches (int): The number of batches to generate.

        Returns:
            new_configs (list): A list of dictionaries containing information about the generated configurations.

        """

        return  generate_new_training_batches(self.generator.model,
                                              n_batches,
                                              self.simulation_topology,
                                              self.config_type_pred_target,
                                              self.init_config_initial_type,
                                              self.device_manager.default_device)


    def __get_config_type(self, batch, type) -> torch.Tensor:
        """
        Function to get a batch of the specified configuration type from the batch.

        Returns:
            config (torch.Tensor): The configurations.

        """

        return get_config_from_batch(batch, type, self.device_manager.default_device)


    def __get_config_type_from_training_batch(self, batch, type) -> torch.Tensor:
        """
        Function to get a batch of the specified configuration type from the training batch.

        Returns:
            config (torch.Tensor): The configurations.

        """

        return get_config_from_training_batch(batch, type, self.device_manager.default_device)


    def __get_loss_avg_p(self, on_last_n_losses) -> float:
        """
        Function for getting the average loss of the predictor model.
        The average loss is calculated on the last n losses.

        If the number of losses is less than n, the average is calculated on all the losses.
        To get the average loss of the last iteration, n should be set to num_training_steps (steps per iteration).

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
        To get the average loss of the last iteration, n should be set to num_training_steps (steps per iteration).

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


    def __get_loss_avg_p_last_iteration(self) -> float:
        """
        Special case of __get_loss_avg_p() where the average is calculated for the last iteration.

        Returns:
            avg_loss_p_last_iteration (float): The average loss of the predictor model on the last iteration.

        """

        return self.__get_loss_avg_p(NUM_TRAINING_STEPS)


    def __get_loss_avg_g_last_iteration(self) -> float:
        """
        Special case of __get_loss_avg_g() where the average is calculated for the last iteration.

        Returns:
            avg_loss_g_last_iteration (float): The average loss of the generator model on the last iteration.

        """

        return self.__get_loss_avg_g(NUM_TRAINING_STEPS)


    def __test_iteration_progress(self) -> dict:
        """
        Function for testing the models.
        The models are tested on the fixed noise.

        Returns:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated targets and predicted targets.

        """

        return test_models(self.generator.model,
                            self.predictor.model,
                            self.simulation_topology,
                            self.init_config_initial_type,
                            self.fixed_input_noise,
                            self.config_type_pred_target,
                            self.device_manager.default_device)


    def __save_progress_plot(self, data) -> None:
        """
        Function for saving the progress plot.
        It save the plot that shows the generated configurations, initial configurations, simulated configurations,
        simulated targets and predicted targets.

        Args:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated targets and predicted targets.

        """

        save_progress_plot(data, self.current_iteration, self.folders.results_folder)


    def __save_models(self) -> None:
        """
        Function for saving the models.

        It saves the generator and predictor models to the models folder every n times they are trained.

        """
        def save(model, path):

            n_times_trained = self.n_times_trained_p if model.type == PREDICTOR else self.n_times_trained_g

            if isinstance(model, nn.DataParallel):
                model_state_dict = model.model.module.state_dict()
            else:
                model_state_dict = model.model.state_dict()

                save_dict = {
                    CHECKPOINT_MODEL_STATE_DICT_KEY       : model_state_dict,
                    CHECKPOINT_MODEL_OPTIMIZER_STATE_DICT : model.optimizer.state_dict(),
                    CHECKPOINT_MODEL_ARCHITECTURE_KEY     : model.model,
                    CHECKPOINT_MODEL_TYPE_KEY             : model.type,
                    CHECKPOINT_MODEL_NAME_KEY             : model.model.name(),
                    CHECKPOINT_ITERATION_KEY              : self.current_iteration,
                    CHECKPOINT_TRAIN_LOSS_KEY             : self.losses[model.type],
                    CHECKPOINT_SEED_KEY                   : self.__seed,
                    CHECKPOINT_DATE_KEY                   : str(datetime.datetime.now()),
                    CHECKPOINT_N_TIMES_TRAINED_KEY        : n_times_trained,
                    CHECKPOINT_P_INPUT_TYPE               : CONFIG_GENERATED,
                    CHECKPOINT_P_TARGET_TYPE              : self.config_type_pred_target
                }

            try:
                torch.save(save_dict, path)
                logging.info(f"Model saved to {path} - iteration: {self.current_iteration+1}")
            except Exception as e:
                logging.error(f"Error saving the model: {e}")


        if self.n_times_trained_p > 0:
            path_p = self.folders.checkpoints_folder / f"predictor_{self.current_iteration+1}.pth.tar"
            save(self.predictor, path_p)

        if self.n_times_trained_g > 0:
            path_g = self.folders.checkpoints_folder / f"generator_{self.current_iteration+1}.pth.tar"
            save(self.generator, path_g)


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
            self.properties_g["can_train"] = self.__get_loss_avg_p_last_iteration() < THRESHOLD_AVG_LOSS_P

            if self.properties_g["can_train"]:
                # +1 because the current iteration is 0-based, +2 because the generator can be trained from the next iteration
                logging.debug(f"Generator can start training from next iteration. The next iteration is number {self.current_iteration + 2}")

        return self.properties_g["can_train"]


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


    def __log_training_step(self, step) -> None:
        """
        Log the progress of the training session inside each iteration.

        Args:
            step (int): The current step in the iteration.

        """
        str_step_time = f"{get_elapsed_time_str(self.step_times_secs[self.current_iteration][step])}"
        str_step      = f"{step+1}/{NUM_TRAINING_STEPS}"
        str_err_p     = f"{self.losses[PREDICTOR][-1]}" if len(self.losses[PREDICTOR]) > 0 else "N/A"
        str_err_g     = f"{self.losses[GENERATOR][-1]}" if len(self.losses[GENERATOR]) > 0 else "N/A"


        with open(self.path_log_file, "a") as log:

            log.write(f"{str_step_time} | Step: {str_step}, Loss P: {str_err_p}, Loss G: {str_err_g}\n")
            log.flush()


    def __log_training_iteration(self) -> None:
        """
        Log the progress of the training session at the end of each iteration.

        """

        with open(self.path_log_file, "a") as log:
            log.write(f"\nElapsed time: {get_elapsed_time_str(self.step_times_secs[self.current_iteration])}\n")

            if self.n_times_trained_p > 0:
                log.write(f"Average loss P: {self.__get_loss_avg_p_last_iteration()}\n")

            if self.n_times_trained_g > 0:
                log.write(f"Average loss G: {self.__get_loss_avg_g_last_iteration()}\n")

            if self.current_iteration + 1 == NUM_ITERATIONS:
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

        balanced_gpu_info = (f"Number of balanced GPU: {self.device_manager.n_balanced_gpus}\n"
                             if self.device_manager.balanced_gpu_indices else "")

        topology_info = ("Topology: toroidal" if self.simulation_topology == TOPOLOGY_TOROIDAL else
                         "Topology: flat" if self.simulation_topology == TOPOLOGY_FLAT else
                         "Topology: unknown")

        generator_info = ""
        if self.properties_g["enabled"]:
            generator_info += (f"Optimizer G: {self.generator.optimizer.__class__.__name__}\n"
                               f"Criterion G: {self.generator.criterion.__class__.__name__}\n")

        log_contents = (
            f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"{seed_info}: {self.__seed}\n"
            f"Default device: {self.device_manager.default_device}\n"
            f"{balanced_gpu_info}\n"
            f"Training specs:\n"
            f"Batch size: {ADV_BATCH_SIZE}\n"
            f"Iterations: {NUM_ITERATIONS}\n"
            f"Number of training steps in each iteration: {NUM_TRAINING_STEPS}\n"
            f"Number of batches generated in each iteration: {NUM_BATCHES} ({NUM_CONFIGS} configs)\n"
            f"Max number of generated batches in dataset: {NUM_MAX_BATCHES} ({NUM_MAX_CONFIGS} configs)\n"
            f"\nSimulation specs:\n"
            f"Grid size: {GRID_SIZE}\n"
            f"Simulation steps: {NUM_SIM_STEPS}\n"
            f"{topology_info}\n"
            f"\nPredicting config type: {self.config_type_pred_target}\n"
            f"\nModel specs:\n"
            f"Optimizer P: {self.predictor.optimizer.__class__.__name__}\n"
            f"Criterion P: {self.predictor.criterion.__class__.__name__}\n"
            f"{generator_info}"
            f"\nTraining progress: \n\n\n"
        )

        with open(path, "w") as log_file:
            log_file.write(log_contents.strip())
            log_file.flush()

        return path


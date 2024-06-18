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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset,\
                             ConcatDataset, Subset


import datetime

from configs.paths import TRAINED_MODELS_DIR, OUTPUTS_DIR
from configs.constants import *

from src.common.folder_manager import FolderManager
from src.common.device_manager import DeviceManager
from src.common.training_base  import TrainingBase
from src.common.model_manager  import ModelManager

from src.common.utils.losses   import WeightedMSELoss, AdversarialGoLLoss

from src.common.generators.binarygen import BinaryGenerator
from src.common.generators.gen       import Gen

from src.common.utils.scores               import prediction_score
from src.common.utils.simulation_functions import simulate_config
from src.common.utils.helpers              import get_elapsed_time_str, \
                                                  get_latest_checkpoint_path, \
                                                  export_figures_to_pdf

from src.gol_adv_sys.utils.helpers import generate_new_batches, \
                                          generate_new_training_batches, \
                                          test_models, \
                                          save_progress_plot, get_config_from_batch,\
                                          save_progress_graph, \
                                          get_config_from_training_batch, \
                                          get_dirichlet_input_noise,\
                                          get_initial_config


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

    def __init__(self,
                 model_p,
                 model_g,
                 target:str,
                 num_sim_steps:int = NUM_SIM_STEPS_DEFAULT) -> None:

        if target in [CONFIG_TARGET_EASY, CONFIG_TARGET_MEDIUM, CONFIG_TARGET_HARD, CONFIG_TARGET_STABLE]:
            self.config_type_pred_target = target
        else:
            raise ValueError("The target type selected is not valid.")

        if num_sim_steps > 0:
            self.num_sim_steps = num_sim_steps
        else:
            raise ValueError("The number of simulation steps must be greater than 0.")

        self.__date = datetime.datetime.now()

        self.__initialize_seed()

        self.folders        = FolderManager(TRAINING_TYPE_ADVERSARIAL, self.__date)

        self.device_manager = DeviceManager()

        self.n_times_trained_p = 0
        self.n_times_trained_g = 0
        self.current_iteration = 0
        self.step_times_secs   = []

        self.complexity_stable_targets= []

        self.losses            = {PREDICTOR: [], GENERATOR: []}
        self.lr_each_iteration = {PREDICTOR: [], GENERATOR: []}

        if model_p.topology != model_g.topology:
            raise ValueError("The topology of the predictor and generator models must be the same.")

        self.simulation_topology = model_p.topology

        self.predictor = ModelManager(model=model_p,
                                      optimizer=optim.SGD(model_p.parameters(),
                                                    lr=P_SGD_LR,
                                                    momentum=P_SGD_MOMENTUM,
                                                    weight_decay=P_SGD_WEIGHT_DECAY),
                                      criterion = AdversarialGoLLoss(model_type=PREDICTOR),
                                      type=PREDICTOR,
                                      device=self.device_manager.primary_device)

        self.generator = ModelManager(model=model_g,
                                      optimizer=optim.AdamW(model_g.parameters(),
                                                    lr=G_ADAMW_LR,
                                                    betas=(G_ADAMW_B1, G_ADAMW_B2),
                                                    eps=G_ADAMW_EPS,
                                                    weight_decay=G_ADAMW_WEIGHT_DECAY),
                                      criterion = AdversarialGoLLoss(model_type=GENERATOR),
                                      type= GENERATOR,
                                      device=self.device_manager.primary_device)

        self.p_num_epochs     = 1
        self.train_dataloader = None

        self.init_config_initial_type = None

        if isinstance(self.generator.model, BinaryGenerator):
            self.init_config_initial_type = INIT_CONFIG_INTIAL_THRESHOLD
        elif isinstance(self.generator.model, Gen):
            self.init_config_initial_type = INIT_CONFIG_INITIAL_SIGN
        else:
            self.init_config_initial_type = INIT_CONFIG_INITIAL_SIGN


        self.fixed_input_noise = get_dirichlet_input_noise(ADV_BATCH_SIZE, device=self.generator.device)

        self.properties_g  = {"enabled": True, "can_train": True}

        self.progress_stats ={"n_cells_initial"  : [],
                              "n_cells_final"    : [],
                              "period"           : [],
                              "transient_phase"  : [],
                              "prediction_score" : [],
                              "iterations"       : []}

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

        self.__update_dataloader()
        self.__warmup_predictor()

        # Empty the dataloader
        self.train_dataloader = None

        for iteration in range(NUM_ITERATIONS):

            self.__update_dataloader()

            logging.info(f"Adversarial training iteration {iteration+1}/{NUM_ITERATIONS}")

            self.step_times_secs.append([])
            self.current_iteration = iteration

            with open(self.path_log_file, "a") as log:
                log_content = (
                    f"\n\nIteration: {iteration+1}/{NUM_ITERATIONS}\n"
                    f"Number of generated configurations in the dataset: {len(self.train_dataloader) * ADV_BATCH_SIZE}\n"
                )

                log.write(log_content)
                log.flush()

            for step in range(NUM_TRAINING_STEPS):

                step_start_time = time.time()

                self.__train_predictor(self.p_num_epochs)

                if self.properties_g["enabled"] and self.properties_g["can_train"]:
                    self.__train_generator()

                step_end_time = time.time()
                self.step_times_secs[self.current_iteration].append(step_end_time - step_start_time)

                self.__log_training_step(step)

            self.__log_training_iteration()

            # Update properties for G
            self.__can_g_train()

            # Test and save models
            data = self.__test_iteration_progress(predictor_device=self.predictor.device,
                                                  generator_device=self.generator.device)
            self.__save_progress_plot(data)
            self.__save_progress_graph()
            self.__save_models()

            self.device_manager.clear_resources()


    def __update_dataloader(self) -> None:
        """
        Get the dataloader for the current iteration.

        Each iteration, a new dataloader is created by adding n_batches new configurations to the dataloader, for a total of
        n_configs new configurations.
        The maximum number of batches in the dataloader is n_max_batches, that contains n_max_configs configurations.
        The older batches of configurations are removed to make room for the new ones.

        The configurations are generated by the generator model.

        """

        chunk     = 32
        times     = NUM_BATCHES // chunk
        remaining = NUM_BATCHES % chunk

        # Placeholder for concatenated datasets
        new_datasets = []

        # Generate configurations in chunks
        for _ in range(times):
            generated, target = self.__get_new_training_batches(chunk, device=self.generator.device)

            generated_cpu = generated.detach().cpu()
            target_cpu    = target.detach().cpu()

            new_datasets.append(TensorDataset(generated_cpu, target_cpu))

            # Clear GPU memory
            del generated, target
            torch.cuda.empty_cache()
            gc.collect()

        # Handle remaining batches
        if remaining > 0:
            generated, target = self.__get_new_training_batches(remaining, device=self.generator.device)

            generated_cpu = generated.detach().cpu()
            target_cpu    = target.detach().cpu()

            new_datasets.append(TensorDataset(generated_cpu, target_cpu))

            # Clear GPU memory
            del generated, target
            torch.cuda.empty_cache()
            gc.collect()

        # Create or update dataloader
        if self.train_dataloader is None:
            combined_dataset      = ConcatDataset(new_datasets)
            self.train_dataloader = DataLoader(combined_dataset, batch_size=ADV_BATCH_SIZE, shuffle=True)
        else:
            # Combine new datasets with the existing dataset
            combined_dataset = ConcatDataset([self.train_dataloader.dataset] + new_datasets)

            total_batches = len(combined_dataset) // ADV_BATCH_SIZE
            if total_batches > NUM_MAX_BATCHES:
                # Trim excess batches
                excess_batches = total_batches - NUM_MAX_BATCHES
                start_idx      = excess_batches * ADV_BATCH_SIZE
                trimmed_dataset = Subset(combined_dataset, range(start_idx, len(combined_dataset)))

                self.train_dataloader = DataLoader(trimmed_dataset, batch_size=ADV_BATCH_SIZE, shuffle=True)
            else:
                self.train_dataloader = DataLoader(combined_dataset, batch_size=ADV_BATCH_SIZE, shuffle=True)

        # Additional garbage collection
        gc.collect()

        logging.debug(f"Updated dataloader with {NUM_BATCHES} new configurations")


    def __warmup_predictor(self) -> None:

        logging.debug(f"Warmup of the predictor model started")

        self.predictor.model.train()


        warmup_values = np.linspace(WARMUP_INITIAL_LR, WARMUP_TARGET_LR, len(self.train_dataloader))

        for batch_count, (generated, target) in enumerate(self.train_dataloader, start=1):

            self.predictor.set_learning_rate(warmup_values[batch_count-1])

            logging.debug(f"Warmup phase => Predictor learning rate: {self.predictor.get_learning_rate()}")

            self.predictor.optimizer.zero_grad()

            predicted = self.predictor.model(generated.detach().to(self.predictor.device))
            errP      = self.predictor.criterion(predicted, target.detach().to(self.predictor.device))

            errP.backward()
            self.predictor.optimizer.step()

        logging.debug(f"Warmup phase => ended")


    def __train_predictor(self, num_epochs: int) -> float:
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

                predicted = self.predictor.model(generated.to(self.predictor.device))
                errP = self.predictor.criterion(predicted, target.to(self.predictor.device))

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

        n = len(self.train_dataloader) * self.p_num_epochs

        for batch_count in range(1, n+1):

            self.generator.optimizer.zero_grad()

            generated, target = self.__get_new_training_batches(n_batches=1, device=self.generator.device)
            predicted = self.predictor.model(generated.to(self.predictor.device))
            errG      = self.generator.criterion(predicted, target.to(self.predictor.device))

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
                                     self.num_sim_steps,
                                     self.init_config_initial_type,
                                     self.device_manager.default_device)


    def __get_new_training_batches(self, n_batches, device) -> Tuple[torch.Tensor, torch.Tensor]:
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
                                              self.num_sim_steps,
                                              self.init_config_initial_type,
                                              device)


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


    def __test_iteration_progress(self,
                                  predictor_device,
                                  generator_device) -> dict:
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
                           self.num_sim_steps,
                           predictor_device,
                           generator_device)


    def __save_progress_plot(self, data) -> None:
        """
        Function for saving the progress plot.
        It save the plot that shows the generated configurations, initial configurations, simulated configurations,
        simulated targets and predicted targets.

        Args:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated targets and predicted targets.

        """

        save_progress_plot(data, self.current_iteration, self.num_sim_steps, self.folders.results_folder)


    def __save_progress_graph(self) -> None:
        """
        Function for showing the progress graph.
        It shows the average number of cells in the initial and final configurations, the period and the transient phase
        of the simulated configurations and the prediction score.

        """

        # Test the models on the fixed noise
        with torch.no_grad():
            self.generator.model.eval()
            self.predictor.model.eval()

            n_configs = ADV_BATCH_SIZE * 16 # 1024 if ADV_BATCH_SIZE = 64
            n_batches = n_configs // ADV_BATCH_SIZE
            generated_config, target = self.__get_new_training_batches(n_batches, self.generator.device)

            initial_config   = get_initial_config(generated_config, self.init_config_initial_type)
            predicted_config = self.predictor.model(generated_config.detach().to(self.predictor.device))

            sim_results = simulate_config(config=initial_config,
                                          topology=self.simulation_topology,
                                          steps=self.num_sim_steps,

                                          device=self.generator.device)

            k = np.random.choice(n_configs-1, 5, replace=False)
            if True:
                fig, axs = plt.subplots(5, 4, figsize=(8, 10))
                for i in range(5):
                    imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}

                    titles = ["initial", "simulated", "final", "metadata"]

                    axs[i,0].imshow(sim_results["initial"][k[i]].cpu().numpy().squeeze(), **imshow_kwargs)
                    axs[i,0].set_title(titles[0]+f"conf - {k[i]}")
                    axs[i,0].axis('off')

                    axs[i,1].imshow(sim_results["simulated"][k[i]].cpu().numpy().squeeze(), **imshow_kwargs)
                    axs[i,1].set_title(titles[1]+f"conf - {k[i]}")
                    axs[i,1].axis('off')

                    axs[i,2].imshow(sim_results["final"][k[i]].cpu().numpy().squeeze(), **imshow_kwargs)
                    axs[i,2].set_title(titles[2]+f"conf - {k[i]}")
                    axs[i,2].axis('off')

                    axs[i,3].text(0.5, 0.5, f"Period: {sim_results['period'][i].item()}", fontsize=12, ha='center', va='center')
                    axs[i,3].text(0.5, 0.3, f"Transient phase: {sim_results['transient_phase'][i].item()}", fontsize=12, ha='center', va='center')
                    axs[i,3].axis('off')

                # Save and close
                pdf_path = self.folders.base_folder / f"iter{self.current_iteration+1}_check_config.pdf"
                export_figures_to_pdf(pdf_path, fig)

            p_score = prediction_score(predicted_config, target)

            # get avgs
            n_intial_cells = sim_results["n_cells_initial"].float().mean().item()
            n_final_cells  = sim_results["n_cells_final"].float().mean().item()
            period         = sim_results["period"].float().mean().item()
            transient_phase= sim_results["transient_phase"].float().mean().item()

            self.progress_stats["n_cells_initial"].append(n_intial_cells)
            self.progress_stats["n_cells_final"].append(n_final_cells)
            self.progress_stats["period"].append(period)
            self.progress_stats["transient_phase"].append(transient_phase)
            self.progress_stats["prediction_score"].append(p_score*100)
            self.progress_stats["iterations"].append(self.current_iteration+1)

            save_progress_graph(self.progress_stats, self.folders.base_folder)


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
                             if self.device_manager.balanced_gpu_indices else "No balanced GPUs available")

        topology_info = ("Topology              : TOROIDAL" if self.simulation_topology == TOPOLOGY_TOROIDAL else
                         "Topology              : FLAT" if self.simulation_topology == TOPOLOGY_FLAT else
                         "Topology              : UNKNOWN")

        generator_info = ""
        if self.properties_g["enabled"]:
            generator_info += (f"Optimizer G: {self.generator.optimizer.__class__.__name__}\n"
                               f"Criterion G: {self.generator.criterion.__class__.__name__}\n"
                               f"Name G     : {self.generator.model.name()}\n"
                               f"Device G   : {self.generator.device}")

        predictor_info = (f"Optimizer P: {self.predictor.optimizer.__class__.__name__}\n"
                          f"Criterion P: {self.predictor.criterion.__class__.__name__}\n"
                          f"Name P     : {self.predictor.model.name()}\n"
                          f"Device P   : {self.predictor.device}")


        init_config_initial_type_info = ""
        if self.init_config_initial_type == INIT_CONFIG_INTIAL_THRESHOLD:
            init_config_initial_type_info = "Initialization initial: THRESHOLD"
        elif self.init_config_initial_type == INIT_CONFIG_INITIAL_SIGN:
            init_config_initial_type_info = "Initialization initial: SIGN"
        elif self.init_config_initial_type == INIT_CONFIG_INITAL_N_CELLS:
            init_config_initial_type_info = "Initialization initial: N_CELLS"
        else:
            init_config_initial_type_info = "Initialization initial: UNKNOWN"

        log_contents = (
            f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"{seed_info}: {self.__seed}\n"
            f"Default device: {self.device_manager.default_device}\n"
            f"{balanced_gpu_info}\n"
            f"\n-----------TRAINING-----------\n"
            f"Batch size: {ADV_BATCH_SIZE}\n"
            f"Iterations: {NUM_ITERATIONS}\n"
            f"Number of training steps in each iteration   : {NUM_TRAINING_STEPS}\n"
            f"Number of batches generated in each iteration: {NUM_BATCHES} ({NUM_CONFIGS} configs)\n"
            f"Max number of generated batches in dataset   : {NUM_MAX_BATCHES} ({NUM_MAX_CONFIGS} configs)\n"
            f"\n----------SIMULATION----------\n"
            f"Grid size             : {GRID_SIZE}\n"
            f"{topology_info}\n"
            f"Simulation steps      : {self.num_sim_steps}\n"
            f"{init_config_initial_type_info}\n"
            f"Target                : {self.config_type_pred_target.upper()}\n"
            f"\n-----------PREDICTOR----------\n"
            f"{predictor_info}\n"
            f"\n-----------GENERATOR----------\n"
            f"{generator_info}\n"
            f"\n-------------------------------------------------------------\n"
            f"\nTraining progress: \n\n\n"
        )

        with open(path, "w") as log_file:
            log_file.write(log_contents.strip())
            log_file.flush()

        return path


    def __check_tensors_device(self):
        """
        Check the device of the tensors in the memory.

        """

        logging.debug("Checking tensors device")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    logging.debug(f"Tensor: {type(obj)}, {obj.size()}, {obj.device}")

            except:
                pass


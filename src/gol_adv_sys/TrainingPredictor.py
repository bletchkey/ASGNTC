"""
This module contains the TrainingPredictor class.

This class is used to train the predictor model on the dataset.

"""
import numpy as np
import random
import time
import datetime
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from config.paths import DATASET_DIR, TRAINED_MODELS_DIR
from config.constants import *

from src.gol_adv_sys.FolderManager import FolderManager
from src.gol_adv_sys.DeviceManager import DeviceManager
from src.gol_adv_sys.DatasetManager import FixedDataset, PairedDataset
from src.gol_adv_sys.ModelManager import ModelManager
from src.gol_adv_sys.TrainingBase import TrainingBase

from src.gol_adv_sys.utils.losses import weigthed_mse_loss, cross_entropy_loss
from src.gol_adv_sys.utils.scores import metric_prediction_accuracy

from src.gol_adv_sys.utils.helper_functions import save_progress_plot_dataset, save_losses_plot, \
                                                   test_predictor_model_dataset, get_elapsed_time_str, \
                                                   get_config_from_batch


class TrainingPredictor(TrainingBase):
    """
    Class designed to handle the training of the predictor model on the dataset.

    Attributes:
        __date (datetime): The date and time when the training session was started.
        __seed_type (dict): The type of seed used for the random number generators.
        __seed (int): The seed used for the random number generators.
        __folders (FolderManager): The folder manager used for managing the folders used in the training session.
        device_manager (DeviceManager): The device manager used for managing the devices used in the training session.
        predictor (ModelManager): The model manager used for managing the predictor model.
        simulation_topology (str): The topology used for the simulation.
        init_config_initial_type (str): The type of initial configuration used for the simulation.
        metric_type (str): The type of metric used for the simulation.
        current_epoch (int): The current epoch of the training session.
        step_times_secs (list): The times in seconds for each step of the training session.
        losses (dict): The losses of the predictor model during the training session.
        accuracy (list): The accuracy of the predictor model during the training session.
        learning_rates (list): The learning rates used during the training session.
        dataset (dict): The dataset used for the training session.
        dataloader (dict): The dataloaders used for the training session.
        path_log_file (str): The path to the log file used for the training session.

    """
    def __init__(self, model=None) -> None:

        self.__date = datetime.datetime.now()
        self.__initialize_seed()
        self.__folders = FolderManager(self.__date)

        self.device_manager = DeviceManager()

        optimizer = optim.SGD(model.parameters(),
                              lr=P_SGD_LR,
                              momentum=P_SGD_MOMENTUM,
                              weight_decay=P_SGD_WEIGHT_DECAY)

        self.predictor = ModelManager(model=model, optimizer=optimizer, criterion=weigthed_mse_loss, device_manager=self.device_manager)

        self.simulation_topology = TOPOLOGY_TOROIDAL
        self.init_config_initial_type = INIT_CONFIG_INTIAL_THRESHOLD
        self.metric_type = CONFIG_METRIC_STABLE

        self.current_epoch = 0
        self.n_times_trained_p = 0
        self.step_times_secs = []

        self.learning_rates = []

        self.losses = {"predictor_train": [],
                       "predictor_val": [],
                       "predictor_test": []}

        self.accuracies = []

        self.dataset = {"train": None, "train_meta": None,
                        "val": None, "val_meta": None,
                        "test": None, "test_meta": None}

        self.dataloader = {"train": None, "val": None, "test": None}

        self.path_log_file = self.__init_log_file()


    def run(self):
        """
        Function used for running the training session.

        """
        self.__init_data()

        self._fit()


    def _fit(self):

        logging.info(f"Training started at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        """
        Training loop for the predictor model.

        The predictor model is trained on the dataset.

        """
        torch.autograd.set_detect_anomaly(True)

        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.predictor.optimizer, mode="min", factor=0.1,
            patience=2, verbose=True, threshold=1e-4,
            threshold_mode="rel", cooldown=2, min_lr=0, eps=1e-8
        )

        # Warmup scheduler
        warmup_lr_values = np.linspace(WARMUP_INITIAL_LR, WARMUP_TARGET_LR, WARMUP_TOTAL_STEPS).tolist()

        total_steps = 0

        self.predictor.set_learning_rate(WARMUP_INITIAL_LR)

        # Training loop
        for epoch in range(NUM_EPOCHS):

            self.current_epoch = epoch
            self.learning_rates.append(self.predictor.get_learning_rate())

            logging.debug(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
            logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

            epoch_start_time = time.time()

            # Train the predictor model
            train_loss = 0
            self.predictor.model.train()
            for batch, _ in self.dataloader["train"]:
                self.predictor.optimizer.zero_grad()
                predicted_metric = self.predictor.model(self.__get_initial_config(batch))
                errP = self.predictor.criterion(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                errP.backward()
                self.predictor.optimizer.step()
                train_loss += errP.item()
                total_steps += 1
                logging.debug(f"Batches processed: {total_steps}")

                # Updates during the warm-up phase
                if total_steps <= WARMUP_TOTAL_STEPS:
                    self.predictor.set_learning_rate(warmup_lr_values[total_steps-1])
                    logging.debug(f"Warm-up phase")
                    logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

            self.n_times_trained_p += 1
            train_loss /= len(self.dataloader["train"])
            self.losses["predictor_train"].append(train_loss)
            logging.debug(f"Predictor loss on train data: {train_loss}")

            # Check the validation loss
            val_loss = 0
            accuracy_avg = 0
            self.predictor.model.eval()
            with torch.no_grad():
                for batch, _ in self.dataloader["val"]:
                    predicted_metric = self.predictor.model(self.__get_initial_config(batch))
                    errP = self.predictor.criterion(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                    val_loss += errP.item()

                    accuracy = metric_prediction_accuracy(self.__get_metric_config(batch, self.metric_type), predicted_metric)
                    accuracy = accuracy.mean().item()
                    accuracy_avg += accuracy
                    logging.debug(f"Accuracy on validation batch: {accuracy}")

            # Compute metric prediction accuracy
            accuracy_avg /= len(self.dataloader["val"])
            logging.debug(f"Accuracy: {accuracy_avg}")
            self.accuracies.append(accuracy_avg)

            val_loss /= len(self.dataloader["val"])
            self.losses["predictor_val"].append(val_loss)
            logging.debug(f"Predictor loss on validation data: {val_loss}")

            # Update the learning rate
            if total_steps > WARMUP_TOTAL_STEPS:
                lr_scheduler.step(val_loss)
                logging.debug(f"Reduce On Plateau phase")
                logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            logging.debug(f"Epoch elapsed time: {get_elapsed_time_str(epoch_elapsed_time)}")

            # Log the training epoch progress
            self.__log_training_epoch(epoch_elapsed_time)

            # Test the predictor model
            logging.debug(f"Testing Predictor model")
            data = self.__test_predictor_model()
            self.__save_progress_plot(data)
            self.__save_losses_plot()

            # Save the model every 10 epochs
            if (epoch > 0) and (epoch % 10 == 0) and (self.n_times_trained_p > 0):
                path = self.__folders.models_folder / f"predictor_{epoch}.pth.tar"
                self.predictor.save(path)
                logging.debug(f"Predictor model saved at {path}")

            self.device_manager.clear_resources()

        # Check the test loss
        test_loss = 0
        self.predictor.model.eval()
        with torch.no_grad():
            for batch, _ in self.dataloader["test"]:
                predicted_metric = self.predictor.model(self.__get_initial_config(batch))
                errP = self.predictor.criterion(predicted_metric, self.__get_metric_config(batch, self.metric_type))
                test_loss += errP.item()

        test_loss /= len(self.dataloader["test"])
        self.losses["predictor_test"].append(test_loss)
        logging.debug(f"Predictor loss on test data: {test_loss}")


        str_err_p_test = f"{self.losses['predictor_test'][-1]}"

        with open(self.path_log_file, "a") as log:
            log.write("\n\nPerformance of the predictor model on the test set:\n")
            log.write(f"Loss P (test): {str_err_p_test}\n\n")
            log.write(f"\n\nTraining ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            log.write(f"Number of times P was trained: {self.n_times_trained_p}\n")
            log.flush()

        logging.info(f"Training ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


    def __init_data(self) -> bool:
        try:
            train_path = DATASET_DIR / f"{DATASET_NAME}_train.pt"
            val_path   = DATASET_DIR / f"{DATASET_NAME}_val.pt"
            test_path  = DATASET_DIR / f"{DATASET_NAME}_test.pt"

            train_meta_path = DATASET_DIR / f"{DATASET_NAME}_metadata_train.pt"
            val_meta_path   = DATASET_DIR / f"{DATASET_NAME}_metadata_val.pt"
            test_meta_path  = DATASET_DIR / f"{DATASET_NAME}_metadata_test.pt"

            self.dataset["train"] = FixedDataset(train_path)
            self.dataset["val"]   = FixedDataset(val_path)
            self.dataset["test"]  = FixedDataset(test_path)

            self.dataset["train_meta"] = FixedDataset(train_meta_path)
            self.dataset["val_meta"]   = FixedDataset(val_meta_path)
            self.dataset["test_meta"]  = FixedDataset(test_meta_path)

            train_ds = PairedDataset(self.dataset["train"], self.dataset["train_meta"])
            val_ds   = PairedDataset(self.dataset["val"], self.dataset["val_meta"])
            test_ds  = PairedDataset(self.dataset["test"], self.dataset["test_meta"])

            self.dataloader["train"] = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            self.dataloader["val"]   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            self.dataloader["test"]  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        except Exception as e:
            logging.error(f"Error initializing the data: {e}")
            raise e

        return True


    def __log_training_epoch(self, time):
        """
        Log the progress of the training session inside each epoch for the predictor model.

        """
        with open(self.path_log_file, "a") as log:

            str_epoch_time  = f"{get_elapsed_time_str(time)}"
            str_epoch       = f"{self.current_epoch+1}/{NUM_EPOCHS}"
            str_err_p_train = f"{self.losses['predictor_train'][-1]:.6f}"
            str_err_p_val   = f"{self.losses['predictor_val'][-1]:.6f}"
            str_accuracy    = f"{(100*self.accuracies[-1]):.1f}%"

            lr = self.learning_rates[self.current_epoch]

            log.write(f"{str_epoch_time} | Epoch: {str_epoch}| Loss P (train): {str_err_p_train}| "
                      f"Loss P (val): {str_err_p_val}| Accuracy: {str_accuracy} | Learning rate: {lr}\n"
            )
            log.flush()


    def __init_log_file(self):
        """
        Create a log file for the training session and write the initial specifications.

        Returns:
            path (str): The path to the log file.
        """
        path = self.__folders.logs_folder / "training.txt"
        seed_info = "Random seed" if self.__seed_type["random"] else \
                    "Fixed seed" if self.__seed_type["fixed"] else "Unknown seed"

        balanced_gpu_info = (f"Balanced GPU indices: {self.device_manager.balanced_gpu_indices}\n"
                             if len(self.device_manager.balanced_gpu_indices) > 0 else "")

        log_contents = (
            f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"{seed_info}: {self.__seed}\n"
            f"Default device: {self.device_manager.default_device}\n"
            f"{balanced_gpu_info}\n"
            f"Training specs:\n"
            f"Batch size: {BATCH_SIZE}\n"
            f"Epochs: {NUM_EPOCHS}\n"
            f"Predicting metric type: {self.metric_type}\n\n"
            f"Training progress:\n\n"
        )

        with open(path, "w") as log_file:
            log_file.write(log_contents)
            log_file.flush()

        return path


    def __get_initial_config(self, batch):
        """
        Function to get a batch of initial configurations from the batch.

        """

        return get_config_from_batch(batch, CONFIG_INITIAL, self.device_manager.default_device)


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
        return test_predictor_model_dataset(self.dataloader["test"],
                                            self.metric_type, self.predictor.model,
                                            self.device_manager.default_device)


    def __save_progress_plot(self, data):
        """
        Function for saving the progress plot.
        It save the plot that shows the generated configurations, initial configurations, simulated configurations,
        simulated metrics and predicted metrics.

        Args:
            data (dict): Contains the generated configurations, initial configurations, simulated configurations,
            simulated metrics and predicted metrics.

        """
        save_progress_plot_dataset(data, self.current_epoch, self.__folders.results_folder)


    def __save_losses_plot(self):
        """
        Function for plotting and saving the losses of training and validation for the predictor model.

        """
        save_losses_plot(self.losses["predictor_train"], self.losses["predictor_val"],
                         self.learning_rates, self.__folders.base_folder)


    def __initialize_seed(self):
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


    # TODO: This function needs to be implemented properly
    def load_pretrained_model(self, model_name: str) -> None:
     """
     Function for loading a model from a file.

     Args:
         name_p (str): The name of the predictor model to load.

     """

     path = TRAINED_MODELS_DIR / model_name

     if path.is_file():
        checkpoint = torch.load(path)
        self.predictor.model.load_state_dict(checkpoint["state_dict"])
        self.predictor.optimizer.load_state_dict(checkpoint["optimizer"])


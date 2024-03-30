"""
This module contains the TrainingPredictor class.

This class is used to train the predictor model on the dataset.

"""

import numpy as np
import random
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from   pathlib import Path
from   torch.utils.data import DataLoader, SubsetRandomSampler

from configs.paths     import DATASET_DIR, TRAINED_MODELS_DIR
from configs.constants import *

from src.common.folder_manager  import FolderManager
from src.common.device_manager  import DeviceManager
from src.common.model_manager   import ModelManager
from src.common.training_base   import TrainingBase

from src.gol_pred_sys.dataset_manager import FixedDataset, PairedDataset

from src.common.utils.losses import WeightedMSELoss, WeightedBCELoss, CustomGoLLoss
from src.common.utils.scores import config_prediction_accuracy

from src.common.utils.helpers import get_elapsed_time_str

from src.gol_pred_sys.utils.helpers import test_predictor_model_dataset, save_progress_plot_dataset, \
                                           save_loss_acc_plot, get_config_from_batch


class TrainingPredictor(TrainingBase):
    """
    Class designed to handle the training of the predictor model on the dataset.

    Attributes:
        __date (datetime): The date and time when the training session was started.
        __seed_type (dict): The type of seed used for the random number generators.
        __seed (int): The seed used for the random number generators.
        __folders (FolderManager): The folder manager object used to manage the folders for the training session.
        device_manager (DeviceManager): The device manager object used to manage the devices for the training session.
        predictor (ModelManager): The model manager object used to manage the predictor model.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler for the predictor model.
        warmup_phase (dict): The warm-up phase specifications.
        config_type_pred_input (str): The configuration type used as input for the predictor model (tipically the initial).
        config_type_pred_target (str): The configuration type used as target for the predictor model (tipically the metric).
        current_epoch (int): The current epoch of the training session.
        n_times_trained_p (int): The number of times the predictor model was trained.
        learning_rates (list): The learning rates used during the training session.
        losses (dict): The losses of the predictor model on the training, validation and test sets.
        accuracies (dict): The accuracies of the predictor model on the training, validation and test sets.
        dataset (dict): The dataset used for training the predictor model.
        dataloader (dict): The dataloaders used for training, validation and testing the predictor model.
        path_log_file (str): The path to the log file for the training session.

    """

    def __init__(self, model=None) -> None:

        self.__date = datetime.datetime.now()
        self.__initialize_seed()
        self.__folders = FolderManager(self.__date)

        self.device_manager = DeviceManager()

        self.predictor = ModelManager(model=model,
                                      optimizer=optim.SGD(model.parameters(),
                                                    lr=P_SGD_LR,
                                                    momentum=P_SGD_MOMENTUM,
                                                    weight_decay=P_SGD_WEIGHT_DECAY),
                                      criterion=CustomGoLLoss(),
                                      device_manager=self.device_manager)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                self.predictor.optimizer, mode="min", factor=0.1,
                                patience=2, verbose=True, threshold=1e-4,
                                threshold_mode="rel", cooldown=2, min_lr=0, eps=1e-8)

        self.warmup_phase = {"enabled": True,
                             "values": np.linspace(WARMUP_INITIAL_LR,
                                                WARMUP_TARGET_LR,
                                                WARMUP_TOTAL_STEPS).tolist()}

        self.config_type_pred_input  = CONFIG_INITIAL
        self.config_type_pred_target = CONFIG_METRIC_MEDIUM

        self.current_epoch = 0
        self.n_times_trained_p = 0

        self.learning_rates = []

        self.losses     = {TRAIN: [], VALIDATION: [], TEST: []}
        self.accuracies = {TRAIN: [], VALIDATION: [], TEST: []}

        self.dataset = {TRAIN: None,
                        TRAIN_METADATA: None,
                        VALIDATION: None,
                        VALIDATION_METADATA: None,
                        TEST: None,
                        TEST_METADATA: None}

        self.dataloader = {TRAIN: None, VALIDATION: None, TEST: None}

        self.path_log_file = self.__init_log_file()


    def run(self) -> None:
        """
        Function used for running the training session.

        """

        torch.autograd.set_detect_anomaly(True)

        logging.info(f"Training started at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        self.__init_data()
        self._fit()

        logging.info(f"Training ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


    def _fit(self) -> None:
        """
        Training loop for training the predictor model on the dataset.

        The Warm-up phase is used to gradually increase the learning rate from WARMUP_INITIAL_LR to WARMUP_TARGET_LR.
        The Warm-up phase lasts for WARMUP_TOTAL_STEPS steps.

        The training loop consists of NUM_EPOCHS epochs.

        """

        if self.warmup_phase["enabled"] == True:
            logging.debug(f"Warm-up phase enabled")
            logging.debug(f"Warm-up phase duration: {WARMUP_TOTAL_STEPS} steps")

            self.predictor.set_learning_rate(WARMUP_INITIAL_LR)

        logging.debug(f"Starting the training loop for the predictor model")
        for epoch in range(NUM_EPOCHS):

            self.current_epoch = epoch
            self.learning_rates.append(self.predictor.get_learning_rate())

            logging.debug(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
            logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

            epoch_start_time = time.time()

            self.process_predictor_model(TRAIN)
            self.process_predictor_model(VALIDATION)

            if self.current_epoch+1 == NUM_EPOCHS:
                self.process_predictor_model(TEST)

            if self.current_epoch > 0:
                self.lr_scheduler.step(self.losses[VALIDATION][-1])
                logging.debug(f"Reduce On Plateau phase")
                logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

            epoch_end_time     = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            logging.debug(f"Epoch elapsed time: {get_elapsed_time_str(epoch_elapsed_time)}")

            self.__log_training_epoch(epoch_elapsed_time)

            # Test the predictor model
            logging.debug(f"Plotting the progress of the predictor model using the test set")
            data = self.__test_predictor_model()
            self.__save_progress_plot(data)
            self.__save_loss_acc_plot()

            # Save the model every 10 epochs
            if (epoch > 0) and (epoch % 10 == 0) and (self.n_times_trained_p > 0):
                path = self.__folders.models_folder / f"predictor_{epoch}.pth.tar"
                self.predictor.save(path)
                logging.debug(f"Predictor model saved at {path}")

            self.device_manager.clear_resources()


    def process_predictor_model(self, mode) -> None:
        """
        Process the predictor model on the specified DataLoader.

        If the mode is TRAIN, the model is trained on the training data.
        If the mode is VALIDATION, the model is tested on the validation data.
        If the mode is TEST, the model is tested on the test data.

        The test on the test data is only performed after the last epoch.

        The test on the validation data is used to update the learning rate,
        check for problems like overfitting, etc.

        Args:
            mode (str): The mode of the processing.

        Raises:
            ValueError: If an invalid mode is passed.

        """

        if mode not in [TRAIN, VALIDATION, TEST]:
            raise ValueError(f"Invalid mode: {mode}, must be one of '{TRAIN}', '{VALIDATION}', '{TEST}'")

        total_loss     = 0
        total_accuracy = 0
        dataloader     = self.dataloader[mode]

        if mode == TRAIN:
            self.predictor.model.train()
        else:
            self.predictor.model.eval()

        process_context = torch.enable_grad() if mode == TRAIN else torch.no_grad()

        with process_context:
            for batch_count, (batch, _) in enumerate(dataloader, start=1):
                if mode == TRAIN:
                    logging.debug(f"Processing batch {batch_count} of epoch {self.current_epoch+1}/{NUM_EPOCHS}")
                    self.predictor.optimizer.zero_grad()

                predicted = self.predictor.model(self.__get_config_type(batch, self.config_type_pred_input))
                errP      = self.predictor.criterion(predicted, self.__get_config_type(batch, self.config_type_pred_target))

                if mode == TRAIN:
                    errP.backward()
                    self.predictor.optimizer.step()
                    self.n_times_trained_p += 1

                    if self.current_epoch == 0 and self.warmup_phase.get("enabled", False):
                        self.predictor.set_learning_rate(self.warmup_phase["values"][batch_count-1])
                        logging.debug("Warm-up phase")
                        logging.debug(f"Learning rate: {self.predictor.get_learning_rate()}")

                accuracy = config_prediction_accuracy(predicted, self.__get_config_type(batch, self.config_type_pred_target))

                total_loss     += errP.item()
                total_accuracy += accuracy

                running_avg_loss     = total_loss / batch_count
                running_avg_accuracy = total_accuracy / batch_count

        self.losses[mode].append(running_avg_loss)
        self.accuracies[mode].append(running_avg_accuracy)

        logging.debug(f"Predictor loss on {mode} data: {self.losses[mode][-1]}")
        logging.debug(f"Accuracy on {mode} data: {self.accuracies[mode][-1]}")


    def __init_data(self) -> bool:
        """
        Initialize the dataset and dataloaders for the training session.

        Returns:
            success (bool): True if the data was initialized successfully, False otherwise.

        """

        try:
            train_path = DATASET_DIR / f"{DATASET_NAME}_train.pt"
            val_path   = DATASET_DIR / f"{DATASET_NAME}_val.pt"
            test_path  = DATASET_DIR / f"{DATASET_NAME}_test.pt"

            train_meta_path = DATASET_DIR / f"{DATASET_NAME}_metadata_train.pt"
            val_meta_path   = DATASET_DIR / f"{DATASET_NAME}_metadata_val.pt"
            test_meta_path  = DATASET_DIR / f"{DATASET_NAME}_metadata_test.pt"

            self.dataset[TRAIN]      = FixedDataset(train_path)
            self.dataset[VALIDATION] = FixedDataset(val_path)
            self.dataset[TEST]       = FixedDataset(test_path)

            self.dataset[TRAIN_METADATA]      = FixedDataset(train_meta_path)
            self.dataset[VALIDATION_METADATA] = FixedDataset(val_meta_path)
            self.dataset[TEST_METADATA]       = FixedDataset(test_meta_path)

            train_ds = PairedDataset(self.dataset[TRAIN], self.dataset[TRAIN_METADATA])
            val_ds   = PairedDataset(self.dataset[VALIDATION], self.dataset[VALIDATION_METADATA])
            test_ds  = PairedDataset(self.dataset[TEST], self.dataset[TEST_METADATA])

            self.dataloader[TRAIN] = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            self.dataloader[VALIDATION]   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            self.dataloader[TEST]  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        except Exception as e:
            logging.error(f"Error initializing the data: {e}")
            raise e

        return True


    def __log_training_epoch(self, time) -> None:
        """
        Log the progress of the training session inside each epoch for the predictor model.

        Args:
            time (float): The elapsed time of the epoch.

        """

        str_epoch_time  = f"{get_elapsed_time_str(time)}"
        str_epoch       = f"{self.current_epoch+1}/{NUM_EPOCHS}"
        str_err_p_train = f"{self.losses[TRAIN][-1]:.6f}"
        str_err_p_val   = f"{self.losses[VALIDATION][-1]:.6f}"
        str_acc_p_train = f"{(100*self.accuracies[TRAIN][-1]):.1f}%"
        str_acc_p_val   = f"{(100*self.accuracies[VALIDATION][-1]):.1f}%"
        str_losses      = f"Losses P [train: {str_err_p_train}, val: {str_err_p_val}]"
        str_accuracies  = f"Accuracies P [train: {str_acc_p_train}, val: {str_acc_p_val}]"
        lr              = self.learning_rates[self.current_epoch]

        with open(self.path_log_file, "a") as log:
            log.write(f"{str_epoch_time} | Epoch: {str_epoch} | {str_losses} | {str_accuracies} | LR: {lr}\n")

            if self.current_epoch+1 == NUM_EPOCHS:
                str_err_p_test  = f"{self.losses[TEST][-1]:.6f}"
                str_acc_p_test  = f"{(100*self.accuracies[TEST][-1]):.1f}%"

                log.write(f"\n\n")
                log.write(f"Performance of the predictor model on the test set:\n")
                log.write(f"Loss P [test: {str_err_p_test}]\n")
                log.write(f"Accuracy P [test: {str_acc_p_test}]\n\n")
                log.write(f"The predictor model was trained {self.n_times_trained_p} times\n\n")
                log.write(f"Training ended at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

            log.flush()


    def __init_log_file(self) -> str:
        """
        Create a log file for the training session and write the initial specifications.

        Returns:
            path (str): The path to the log file.

        """

        path = self.__folders.logs_folder / FILE_NAME_TRAINING_PROGRESS
        seed_info = "Random seed" if self.__seed_type["random"] else \
                    "Fixed seed" if self.__seed_type["fixed"] else "Unknown seed"

        balanced_gpu_info = (f"Balanced GPU indices: {self.device_manager.balanced_gpu_indices}\n"
                             if len(self.device_manager.balanced_gpu_indices) > 0 else "")

        log_contents = (
            f"Training session started at {self.__date.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"{seed_info}: {self.__seed}\n"
            f"Default device: {self.device_manager.default_device}\n"
            f"{balanced_gpu_info}\n"
            f"Training specifications:\n\n"
            f"Batch size: {BATCH_SIZE}\n"
            f"Epochs: {NUM_EPOCHS}\n"
            f"Prediction input configuration type: {self.config_type_pred_input}\n"
            f"Prediction target configuration type: {self.config_type_pred_target}\n\n\n"
            f"Training progress:\n\n"
        )

        with open(path, "w") as log_file:
            log_file.write(log_contents)
            log_file.flush()

        return path


    def __get_config_type(self, batch, type) -> torch.Tensor:
        """
        Function to get a batch of the specified configuration type from the batch.

        Returns:
            config (torch.Tensor): The configurations.

        """

        return get_config_from_batch(batch, type, self.device_manager.default_device)


    def __test_predictor_model(self) -> dict:
        """
        Function for testing the predictor model.

        Returns:
            data (dict): Contains the results of the test, including the configurations and the predictions.

        """

        return test_predictor_model_dataset(self.dataloader[TEST],
                                            self.config_type_pred_input,
                                            self.config_type_pred_target, self.predictor.model,
                                            self.device_manager.default_device)


    def __save_progress_plot(self, data) -> None:
        """
        Function for saving the progress plot.
        It save the plot that shows a visual representation of the progress of the predictor model.

        Args:
            data (dict): Contains the results of the test, including the configurations and the predictions.

        """

        save_progress_plot_dataset(data, self.current_epoch, self.__folders.results_folder)


    def __save_loss_acc_plot(self) -> None:
        """
        Function for plotting and saving the losses of training and validation for the predictor model.

        """

        save_loss_acc_plot(self.losses[TRAIN], self.losses[VALIDATION],
                           self.accuracies[TRAIN], self.accuracies[VALIDATION],
                           self.learning_rates, self.__folders.base_folder)


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


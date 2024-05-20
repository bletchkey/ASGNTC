import torch
import torch.nn as nn
import torch.optim as optim

from typing import Union, Callable

from src.common.device_manager import DeviceManager
from configs.constants import *


class ModelManager:
    """
    Class for managing the model during training and its hyperparameters.

    Attributes:
        model (nn.Module): The model to be trained.
        optimizer (optim.Optimizer): The optimizer to be used for training.
        criterion (Union[nn.Module, Callable]): The loss function to be used for training.
        device_manager (DeviceManager): The device manager.

    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: Union[nn.Module, Callable], device_manager: DeviceManager):

        self.__model = model
        self.__optimizer = optimizer
        self.__criterion = criterion

        self.__device_manager = device_manager
        self.__model.to(self.__device_manager.default_device)

        # TODO: Implement distributed training with DataParallel (not necessary for now)
        # if self.__device_manager.n_balanced_gpus > 0:
        #     self.__model = nn.DataParallel(self.model, device_ids=self.__device_manager.balanced_gpu_indices)

    @property
    def model(self) -> nn.Module:
        """Returns the model."""

        return self.__model

    @property
    def optimizer(self) -> optim.Optimizer:
        """Returns the optimizer."""
        return self.__optimizer

    @property
    def criterion(self) -> Union[nn.Module, Callable]:
        """Returns the loss function."""
        return self.__criterion


    def save(self, path: str) -> None:
        """
        Function for saving the predictor model.

        It saves predictor model to the models folder every n epochs.

        """

        if isinstance(self.__model, nn.DataParallel):
            torch.save({
                    "state_dict": self.__model.module.state_dict(),
                    "optimizer": self.__optimizer.state_dict(),
                }, path)
        else:
            torch.save({
                    "state_dict": self.__model.state_dict(),
                    "optimizer": self.__optimizer.state_dict(),
                }, path)


    def get_learning_rate(self) -> float:
        """
        Function for getting the learning rate.

        Returns:
            float: The learning rate.

        """

        return self.__optimizer.param_groups[0]["lr"]


    def set_learning_rate(self, lr: float) -> None:
        """
        Function for setting the learning rate.

        Args:
            lr (float): The learning rate.

        """

        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


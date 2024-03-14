import numpy as np

import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from .utils import constants as constants

from .DeviceManager import DeviceManager

from .utils.simulation_functions import simulate_config

class Playground():

    def __init__(self, topology: int):
        self.__topology = topology
        self.__device_manager = DeviceManager()

    @property
    def topology(self) -> int:
        return self.__topology


    def simulate(self, config: torch.Tensor, steps: int, calc_final: bool=True) -> torch.Tensor:

        config = config.to(self.__device_manager.default_device)

        final, metrics, n_cells_init, n_cells_final = simulate_config(config, self.__topology, steps=steps,
                                                        calculate_final_config=calc_final, device=self.__device_manager.default_device)


        results = {
            "period": final["period"],
            "antiperiod": final["antiperiod"],
            "n_cells_init": n_cells_init,
            "n_cells_final": n_cells_final,
            "final_config": final["config"],
            "easy_metric": metrics["easy"],
            "medium_metric": metrics["medium"],
            "hard_metric": metrics["hard"]
        }

        return results



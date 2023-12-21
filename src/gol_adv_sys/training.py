import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import constants as constants

from .model_p import Predictor
from .model_g import Generator

from .train_models import fit

class Training():
    def __init__(self):

        self.__seed_type = {"fixed": 54, "random": random.randint(1, 10000), "is_random": True}
        self.seed = self.__seed_type["random"] if self.__seed_type["is_random"] else self.__seed_type["fixed"]

        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fixed_noise = torch.randn(constants.bs, constants.nz, 1, 1, device=self.device)

        self.criterion_p = nn.MSELoss()
        self.criterion_g = lambda x, y: -1 * nn.MSELoss()(x, y)

        self.model_g = Generator().to(self.device)
        self.model_p = Predictor().to(self.device)

        self.optimizer_g = optim.AdamW(self.model_g.parameters(),
                                       lr=constants.g_adamw_lr,
                                       betas=(constants.g_adamw_b1, constants.g_adamw_b2),
                                       eps=constants.g_adamw_eps,
                                       weight_decay=constants.g_adamw_wd)

        self.optimizer_p = optim.AdamW(self.model_p.parameters(),
                                        lr=constants.p_adamw_lr,
                                        betas=(constants.p_adamw_b1, constants.p_adamw_b2),
                                        eps=constants.p_adamw_eps,
                                        weight_decay=constants.p_adamw_wd)
        self.result = None

    def get_training_specs(self):
        return {
            "seed": self.seed,
            "device": self.device,
            "batch size": constants.bs,
            "epochs": constants.num_epochs,
            "grid size": constants.grid_size,
            "nz": constants.nz,
        }

    def get_device(self):
        return self.device

    def run(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.use_deterministic_algorithms(True) # Needed for reproducible results

        # Fit
        G_losses, P_losses = fit(self.model_g, self.model_p, self.optimizer_g, self.optimizer_p,
                                 self.criterion_g, self.criterion_p, self.fixed_noise, self.device)

        self.result = {
            "G_losses": G_losses,
            "P_losses": P_losses
        }

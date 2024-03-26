import torch
import torch.nn as nn

from config.constants import *


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha: float = 5.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(((target - prediction) ** 2) * self.__weight(target))

        return loss

    def __weight(self, target):
        return 1 + (self.alpha * target)


class WeightedBCELoss(nn.Module):

    def __init__(self, alpha: float = 5.0):
        super(WeightedBCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        loss = torch.mean(-target * torch.log(prediction + eps) -
                    (1 - target) * torch.log(1 - prediction) * self.__weight(target))

        loss += torch.mean(target * torch.log(target + eps) +
                    (1 - target) * torch.log(1 - target) * self.__weight(target))

        return loss

    def __weight(self, target):
        return 1 + (self.alpha * target)


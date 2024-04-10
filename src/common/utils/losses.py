import torch
import torch.nn as nn

from configs.constants import *


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss.

    Args:
        alpha (float): Weighting factor to be applied to the loss

    Returns:
        torch.Tensor: Weighted Mean Squared Error Loss

    """
    def __init__(self, alpha: float = 9.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(((target - prediction) ** 2) * self.__weight(target))

        return loss

    def __weight(self, target):
        return 1 + (self.alpha * target)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.

    Args:
        alpha (float): Weighting factor to be applied to the loss

    Returns:
        torch.Tensor: Weighted Binary Cross Entropy Loss

    """

    def __init__(self, alpha: float = 9.0):
        super(WeightedBCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        prediction = prediction.clamp(min=eps, max=1 - eps)
        target     = target.clamp(min=eps, max=1 - eps)

        weights = self.__weight(target)
        loss = -target * torch.log(prediction) - \
               (1 - target) * torch.log(1 - prediction) + \
                target * torch.log(target) + (1 - target) * torch.log(1 - target) \
                * weights

        return torch.mean(loss)

    def __weight(self, target):
        return 1 + (self.alpha * target)


class CustomGoLLoss(nn.Module):
    """
    Custom loss function for the Game of Life task.

    Args:
        model (str):   Model type
        alpha (float): Weighting factor to be applied to the loss

    Returns:
        torch.Tensor: Custom loss for the Generator/Predictor

    """
    def __init__(self, model:str, alpha: float = 9.0):
        super(CustomGoLLoss, self).__init__()
        self.alpha = alpha
        self.model = model

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                probability: torch.Tensor = None) -> torch.Tensor:

        loss = 0
        weigthed_mse = (target - prediction) ** 2 * self.__weight(target)

        if self.model == PREDICTOR:
            loss = torch.mean(weigthed_mse)

        elif self.model == GENERATOR:
            probability     = probability.view(-1, 1, 1, 1).expand_as(target)
            loss            = torch.mean(probability * weigthed_mse)
        else:
            raise ValueError(f"Invalid model type: {self.model}")

        return loss

    def __weight(self, target):
        return 1 + (self.alpha * target)

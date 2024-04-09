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
        None

    Returns:
        torch.Tensor: Custom Game of Life Loss

    """
    def __init__(self, alpha: float = 9.0):
        super(CustomGoLLoss, self).__init__()
        self.alpha = alpha

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                log_probability: torch.Tensor) -> torch.Tensor:

        log_probability = log_probability.view(-1, 1, 1, 1).expand_as(target)
        weighted_error  = (target - prediction) ** 2 * self.__weight(target)

        loss = -1 * torch.mean(log_probability * weighted_error)

        return loss

    def __weight(self, target):
        return 1 + (self.alpha * target)


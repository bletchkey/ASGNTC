import torch
from config.constants import *


def weigthed_mse_loss(target, prediction):
    """
    Weighted Mean Squared Error Loss

    """
    alpha = 5.0
    loss = torch.mean(((target - prediction) ** 2) * __weight(prediction, alpha)) # * __weight(target, alpha))

    return loss


def cross_entropy_loss(target, prediction):
    """
    Cross Entropy Loss

    """
    alpha = 2.0
    eps = 1e-6

    loss  = torch.mean(-target * torch.log(prediction + eps) - (1 - target) * torch.log(1 - prediction)) * __weight(target, alpha)
    loss += torch.mean(target * torch.log(target + eps) + (1 - target) * torch.log(1 - target)) * __weight(target, alpha)

    return loss


def __weight(target, alpha):
    """
    Weighting function for the output

    """

    return 1 + (alpha * target)


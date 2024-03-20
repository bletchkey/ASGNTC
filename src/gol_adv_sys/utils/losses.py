import torch
from src.gol_adv_sys.utils import constants as constants


def weigthed_mse_loss(target, prediction):
    """
    Weighted Mean Squared Error Loss

    """
    return torch.mean(((target - prediction) ** 2) * __weight(prediction))


def cross_entropy_loss(target, prediction):
    pass


def __weight(prediction):
    """
    Weighting function for the output

    """
    weight_factor = 2.0
    return 1 + (weight_factor * prediction)


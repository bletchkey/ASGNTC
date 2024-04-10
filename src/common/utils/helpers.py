import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from pathlib import Path

from configs.constants import *


def get_elapsed_time_str(times: list) -> str:
    """
    Function to get the elapsed time in an easily readable format
    All the times are summed together and then converted to hours, minutes and seconds

    Args:
        times (list): The list of times in seconds

    Returns:
        time_format (str): The elapsed time in the format "Hh Mm Ss"

    """
    seconds = sum(times) if isinstance(times, list) else times
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60  # Remaining minutes after converting to hours
    remaining_seconds = int(seconds % 60)

    # Format time
    time_format = f"{hours}h {remaining_minutes}m {remaining_seconds}s"

    return time_format


def toroidal_Conv2d(x, conv2d=None, padding=1):
        """
        Function for adding custom padding (toroidal) to a convolutional layer
        """
        x = __add_toroidal_padding(x, padding)
        x = conv2d(x)

        return x


def __add_toroidal_padding(x: torch.Tensor, padding: int=1) -> torch.Tensor:
    """
    Function for adding toroidal padding

    Args:
        x (torch.Tensor): The tensor to add padding to

    Returns:
        x (torch.Tensor): The tensor with toroidal padding

    """
    if x.dim() != 4:
        raise RuntimeError(f"Expected 4D tensor, got {x.dim()}")

    if padding <= 0:
        return x

    x = torch.cat([x[:, :, -padding:], x, x[:, :, :padding]], dim=2)
    x = torch.cat([x[:, :, :, -padding:], x, x[:, :, :, :padding]], dim=3)

    return x


import torch
import torch.nn as nn
import math
from typing import Union

from torchmetrics.functional import structural_similarity_index_measure as ssim

from configs.constants import *


def __calculate_bins_number(tolerance: float) -> int:
    """
    Calculate the number of bins based on the tolerance level.
    The number of bins is calculated as 2^(10*tolerance), with a minimum of 1 and a maximum of 1024.

    Parameters:
        tolerance (float): The tolerance level for considering a prediction as accurate.

    Returns:
        int: The number of bins based on the tolerance level.

    """

    n = math.log2(GRID_SIZE**2)
    num_bins = 2 ** (n * (1-tolerance))
    num_bins = int(num_bins)

    return max(1, min(num_bins, 1024))


def prediction_accuracy(prediction: torch.Tensor, target: torch.Tensor, tolerance: float = 0.1) -> float:

    num_bins = __calculate_bins_number(tolerance)
    min_val, max_val = 0, 1

    # Create bins for histogram
    bins = torch.linspace(min_val, max_val, steps=num_bins + 1)

    target_bins     = torch.bucketize(target, bins)
    prediction_bins = torch.bucketize(prediction, bins)

    correct_predictions = (target_bins == prediction_bins).float()

    mean_accuracy = correct_predictions.mean().item()

    return mean_accuracy


def prediction_accuracy_bins(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the mean accuracy score for the prediction compared to the target.
    The values range from 0 to 1, and are binned into four categories. Scores are awarded based on
    the bin matching, with a perfect match scoring 1, adjacent bins scoring 0.5, and non-adjacent
    bins scoring 0.

    Parameters:
        prediction (torch.Tensor): The prediction tensor with shape (N, C, H, W).
        target (torch.Tensor): The target tensor with shape (N, C, H, W).

    Returns:
        float: The mean accuracy score for the prediction compared to the target.

    Raises:
        ValueError: If the input tensors do not have matching shapes or are not 4-dimensional.

    """
    if target.size() != prediction.size() or target.dim() != 4:
        raise ValueError("Target and prediction tensors must have the same 4-dimensional shape.")

    bin_edges = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=target.device)

    # Flatten the last three dimensions to apply bucketize
    target_flat     = target.contiguous().flatten(start_dim=1)
    prediction_flat = prediction.contiguous().flatten(start_dim=1)

    # Find the indices of the bins to which each value belongs
    target_bins     = torch.bucketize(target_flat, bin_edges, right=True)
    prediction_bins = torch.bucketize(prediction_flat, bin_edges, right=True)

    # Calculate the score
    x = torch.abs(target_bins - prediction_bins)

    score = nn.ReLU()(2 - x.float()) / 2

    mean_scores = score.mean()

    return mean_scores.item()

def prediction_accuracy_tolerance(prediction: torch.Tensor, target: torch.Tensor, tolerance: float) -> float:
    """
    Calculate the accuracy score for the prediction compared to the target based on a tolerance.
    Each predicted value within the specified tolerance of the target value is considered correct.

    Parameters:
        prediction (torch.Tensor): The prediction tensor with shape (N, C, H, W).
        target (torch.Tensor): The target tensor with shape (N, C, H, W).
        tolerance (float): The tolerance level for considering a prediction as accurate.

    Returns:
        float: The mean accuracy score for the prediction compared to the target.

    Raises:
        ValueError: If the input tensors do not have matching shapes or are not 4-dimensional.
    """
    if target.size() != prediction.size() or target.dim() != 4:
        raise ValueError("Target and prediction tensors must have the same 4-dimensional shape.")

    # Flatten the last three dimensions to compute accuracy
    target_flat     = target.contiguous().view(-1)
    prediction_flat = prediction.contiguous().view(-1)

    # Calculate the absolute difference
    difference = torch.abs(prediction_flat - target_flat)

    # Calculate the tolerance based on the target value
    tolerance_values = tolerance * torch.abs(target_flat)

    # Determine which predictions are within the tolerance
    correct_predictions = (difference <= tolerance_values).float()

    # Calculate mean accuracy
    mean_accuracy = correct_predictions.mean().item()

    return mean_accuracy

def prediction_accuracy_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the SSIM-based accuracy score for the prediction compared to the target.
    The Structural Similarity Index (SSIM) measures the visual similarity between two images.

    Parameters:
        prediction (torch.Tensor): The prediction tensor with shape (N, C, H, W).
        target (torch.Tensor): The target tensor with shape (N, C, H, W).

    Returns:
        float: The mean SSIM score for the prediction compared to the target.

    Raises:
        ValueError: If the input tensors do not have matching shapes or are not 4-dimensional.
    """
    if target.size() != prediction.size() or target.dim() != 4:
        raise ValueError("Target and prediction tensors must have the same 4-dimensional shape.")

    # Calculate the SSIM score
    ssim_score = ssim(prediction, target)

    return ssim_score.item()

def calculate_stable_metric_complexity(metrics: torch.Tensor,
                                       mean: bool = False) -> Union[torch.Tensor, float]:
    """
    Calculate the complexity of the stable metric. The complexity is calculated
    by counting the numbers of pixels that are > eps and dividing by the total
    number of pixels.

    Parameters:
        metrics (torch.Tensor): The metrics tensor with shape (N, C, H, W).

    Returns:
        torch.Tensor: The complexity of the stable metrics.

    Raises:
        ValueError: If the input tensor is not 4-dimensional.

    """
    if metrics.dim() != 4:
        raise ValueError("Metrics tensor must have a 4-dimensional shape.")

    eps = 0.05

    complexity = torch.where(metrics > eps,
                             torch.tensor(1, device=metrics.device, dtype=torch.float),
                             torch.tensor(0, device=metrics.device, dtype=torch.float))

    if mean:
        return complexity.mean().item()
    else:
        return complexity.sum(dim=(1, 2, 3)).float() / (metrics.size(2) * metrics.size(3))


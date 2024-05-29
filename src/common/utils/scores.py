import torch
import torch.nn as nn
import math
from typing import Union

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

    return max(1, min(num_bins, GRID_SIZE**2))


def distributions_match(prediction: torch.Tensor, target: torch.Tensor, tolerance: float = 0.5) -> float:

    num_bins = __calculate_bins_number(tolerance)
    min_val, max_val = 0, 1

    # Create bins for histogram
    bins = torch.linspace(min_val, max_val, steps=num_bins + 1, device=target.device)

    # Flatten the last three dimensions to apply bucketize
    target_flat     = target.contiguous().flatten(start_dim=1)
    prediction_flat = prediction.contiguous().flatten(start_dim=1)

    target_bins     = torch.bucketize(target_flat, bins)
    prediction_bins = torch.bucketize(prediction_flat, bins)

    correct_predictions = (target_bins == prediction_bins).float()

    mean_accuracy = correct_predictions.mean().item()

    return mean_accuracy


def prediction_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:

    threshold = 0.05

    predicted_classes = (prediction > threshold).float()
    target_classes    = (target > threshold).float()

    true_positives  = (predicted_classes * target_classes).sum(dim=(1, 2, 3))
    true_negatives  = ((1 - predicted_classes) * (1 - target_classes)).sum(dim=(1, 2, 3))

    positive = target_classes.sum(dim=(1, 2, 3))
    negative = (1 - target_classes).sum(dim=(1, 2, 3))

    accuracy = (true_positives + true_negatives) / (positive + negative)

    return accuracy.mean().item()


def prediction_score(prediction: torch.Tensor, target: torch.Tensor) -> float:

    if (target < 0).any() or (target > 1).any():
        raise ValueError("Target values must be between 0 and 1.")

    if (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("Prediction values must be between 0 and 1.")

    threshold = 0.05

    predicted_classes = (prediction > threshold).float()
    target_classes    = (target > threshold).float()

    # Identify matches, false positives, and false negatives
    matches = (predicted_classes == target_classes).float()
    false_positives = (predicted_classes > target_classes).float()  # Predicted 1 but should be 0
    false_negatives = (predicted_classes < target_classes).float()  # Predicted 0 but should be 1

    # Calculate the total elements per sample in the batch for normalization
    total_elements = matches.size(1) * matches.size(2) * matches.size(3)

    # Calculate match score
    match_score = matches.sum(dim=(1,2,3)) / total_elements

    # Calculate penalties for false positives and false negatives
    fp_penalty = false_positives.sum(dim=(1,2,3)) / total_elements
    fn_penalty = false_negatives.sum(dim=(1,2,3)) / total_elements

    # Normalize penalties
    score = match_score - 0.5 * (fp_penalty + fn_penalty)

    # Clamp the score to ensure it's between 0 and 1
    score = torch.clamp(score, min=0, max=1)

    return score.mean().item()


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


def calculate_stable_target_complexity(targets: torch.Tensor,
                                       mean: bool = False) -> Union[torch.Tensor, float]:
    """
    Calculate the complexity of the stable target. The complexity is calculated
    by counting the numbers of pixels that are > eps and dividing by the total
    number of pixels.

    Parameters:
        targets (torch.Tensor): The targets tensor with shape (N, C, H, W).

    Returns:
        torch.Tensor: The complexity of the stable targets.

    Raises:
        ValueError: If the input tensor is not 4-dimensional.

    """
    if targets.dim() != 4:
        raise ValueError("Targets tensor must have a 4-dimensional shape.")

    eps = 0.05

    complexity = torch.where(targets > eps,
                             torch.tensor(1, device=targets.device, dtype=torch.float),
                             torch.tensor(0, device=targets.device, dtype=torch.float))

    if mean:
        return complexity.mean().item()
    else:
        return complexity.sum(dim=(1, 2, 3)).float() / (targets.size(2) * targets.size(3))


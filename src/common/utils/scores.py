import torch
import torch.nn as nn

from configs.constants import *


def config_prediction_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
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


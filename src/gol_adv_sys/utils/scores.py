import torch
from config.constants import *
import torch
from typing import Tuple


def metric_prediction_accuracy(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
    Calculate the binned accuracy score for each pixel in the prediction compared to the target.
    The values range from 0 to 1, and are binned into four categories. Scores are awarded based on
    the bin matching, with a perfect match scoring 1, adjacent bins scoring 0.5, and non-adjacent
    bins scoring 0.

    Parameters:
        target (torch.Tensor): The target tensor with shape (N, C, H, W).
        prediction (torch.Tensor): The prediction tensor with shape (N, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (N,) where each element is the mean score of the
                      corresponding configuration in the batch.

    Raises:
        ValueError: If the input tensors do not have matching shapes or are not 4-dimensional.

    """
    if target.size() != prediction.size() or target.dim() != 4:
        raise ValueError("Target and prediction tensors must have the same 4-dimensional shape.")

    bin_edges = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=target.device)

    # Flatten the last three dimensions to apply bucketize
    target_flat = target.flatten(start_dim=1)
    prediction_flat = prediction.flatten(start_dim=1)

    # Find the indices of the bins to which each value belongs
    target_bins = torch.bucketize(target_flat, bin_edges, right=True)
    prediction_bins = torch.bucketize(prediction_flat, bin_edges, right=True)

    # Calculate the score
    score = torch.where(target_bins == prediction_bins, 1.0,
                        torch.where(torch.abs(target_bins - prediction_bins) == 1, 0.5, 0.0))

    # Calculate the number of guessed metrics
    guessed = torch.zeros_like(target_flat)
    guessed = torch.where(target_bins == prediction_bins, 1.0, guessed)
    guessed = guessed.sum(dim=1)

    # Compute the mean score for each item in the batch
    mean_scores = score.mean(dim=1)  # Compute mean across the flattened second dimension

    return mean_scores, guessed

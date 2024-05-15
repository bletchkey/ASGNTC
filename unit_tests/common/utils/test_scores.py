import unittest
import torch

from src.common.utils.scores import config_prediction_accuracy, calculate_stable_metric_complexity


class TestConfigPredictionAccuracy(unittest.TestCase):

    def test_accuracy_perfect_match(self):
        prediction = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
        target = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
        accuracy = config_prediction_accuracy(prediction, target)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_accuracy_adjacent_bins(self):
        prediction = torch.tensor([[[[0.1, 0.3], [0.5, 0.7]]]])
        target = torch.tensor([[[[0.2, 0.2], [0.4, 0.8]]]])
        accuracy = config_prediction_accuracy(prediction, target)
        self.assertAlmostEqual(accuracy, 0.625)

    def test_accuracy_non_adjacent_bins(self):
        prediction = torch.tensor([[[[0.0, 0.9], [0.1, 0.8]]], [[[0.0, 0.9], [0.1, 0.8]]]])
        target = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]], [[[0.0, 0.9], [0.1, 0.8]]]])
        accuracy = config_prediction_accuracy(prediction, target)
        self.assertAlmostEqual(accuracy, 0.625)

    def test_input_shape_mismatch(self):
        prediction = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
        target = torch.tensor([[[[0.1, 0.2]]]])  # Incorrect shape
        with self.assertRaises(ValueError):
            config_prediction_accuracy(prediction, target)

    def test_input_dimension_mismatch(self):
        prediction = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Incorrect dimension
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Incorrect dimension
        with self.assertRaises(ValueError):
            config_prediction_accuracy(prediction, target)


class TestCalculateStableMetricComplexity(unittest.TestCase):

    def test_complexity_bs_one_mean_true(self):
        metrics = torch.tensor([[[[1, 0.5], [0, 0]]]])
        complexity = calculate_stable_metric_complexity(metrics, mean=True)
        self.assertAlmostEqual(complexity, 0.5)

    def test_complexity_bs_two_mean_false(self):
        metrics = torch.tensor([[[[1, 0.5], [0, 0]]], [[[0, 0.5], [1, 0]]]])
        complexity = calculate_stable_metric_complexity(metrics, mean=False)
        for c in complexity:
            self.assertAlmostEqual(c, 0.5)

    def test_complexity_bs_two_mean_true(self):
        metrics = torch.tensor([[[[1, 0.5], [0, 0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
        complexity = calculate_stable_metric_complexity(metrics, mean=True)
        self.assertAlmostEqual(complexity, 0.75)

    def test_complexity_bs_two_same_values_mean_true(self):
        metrics = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]], [[[0.5, 0.5], [0.5, 0.5]]]])
        complexity = calculate_stable_metric_complexity(metrics, mean=True)
        self.assertAlmostEqual(complexity, 1.0)

    def test_input_dimension_mismatch(self):
        metrics = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Incorrect dimension
        with self.assertRaises(ValueError):
            calculate_stable_metric_complexity(metrics, mean=True)


if __name__ == '__main__':
    unittest.main()


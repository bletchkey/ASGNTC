import unittest
import torch

from src.common.utils.scores import config_prediction_accuracy


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


if __name__ == '__main__':
    unittest.main()


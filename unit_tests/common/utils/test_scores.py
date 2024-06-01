import unittest
import torch

from src.common.utils.scores import  calculate_stable_target_complexity


class TestCalculateStableTargetComplexity(unittest.TestCase):

    def test_complexity_bs_one_mean_true(self):
        targets = torch.tensor([[[[1, 0.5], [0, 0]]]])
        complexity = calculate_stable_target_complexity(targets, mean=True)
        self.assertAlmostEqual(complexity, 0.5)

    def test_complexity_bs_two_mean_false(self):
        targets = torch.tensor([[[[1, 0.5], [0, 0]]], [[[0, 0.5], [1, 0]]]])
        complexity = calculate_stable_target_complexity(targets, mean=False)
        for c in complexity:
            self.assertAlmostEqual(c, 0.5)

    def test_complexity_bs_two_mean_true(self):
        targets = torch.tensor([[[[1, 0.5], [0, 0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
        complexity = calculate_stable_target_complexity(targets, mean=True)
        self.assertAlmostEqual(complexity, 0.75)

    def test_complexity_bs_two_same_values_mean_true(self):
        targets = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]], [[[0.5, 0.5], [0.5, 0.5]]]])
        complexity = calculate_stable_target_complexity(targets, mean=True)
        self.assertAlmostEqual(complexity, 1.0)

    def test_input_dimension_mismatch(self):
        targets = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Incorrect dimension
        with self.assertRaises(ValueError):
            calculate_stable_target_complexity(targets, mean=True)


if __name__ == '__main__':
    unittest.main()


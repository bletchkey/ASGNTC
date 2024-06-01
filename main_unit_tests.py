import sys
import unittest

from configs.setup     import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths     import CONFIG_DIR

from unit_tests.common.utils.test_scores import TestCalculateStableTargetComplexity


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_ut.json")

    suite = unittest.TestSuite()

    suite.addTest(TestCalculateStableTargetComplexity('test_complexity_bs_one_mean_true'))
    suite.addTest(TestCalculateStableTargetComplexity('test_complexity_bs_two_mean_false'))
    suite.addTest(TestCalculateStableTargetComplexity('test_complexity_bs_two_mean_true'))
    suite.addTest(TestCalculateStableTargetComplexity('test_complexity_bs_two_same_values_mean_true'))
    suite.addTest(TestCalculateStableTargetComplexity('test_input_dimension_mismatch'))

    # Run the suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


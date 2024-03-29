import sys
import unittest

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from unit_tests.common.utils.test_scores import TestConfigPredictionAccuracy, \
                                                TestCalculateStableMetricComplexity


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "unit_tests_logging.json")

    suite = unittest.TestSuite()

    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_perfect_match'))
    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_adjacent_bins'))
    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_non_adjacent_bins'))
    suite.addTest(TestConfigPredictionAccuracy('test_input_shape_mismatch'))
    suite.addTest(TestConfigPredictionAccuracy('test_input_dimension_mismatch'))

    suite.addTest(TestCalculateStableMetricComplexity('test_complexity_bs_one'))
    suite.addTest(TestCalculateStableMetricComplexity('test_complexity_bs_two'))
    suite.addTest(TestCalculateStableMetricComplexity('test_input_dimension_mismatch'))

    # Run the suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


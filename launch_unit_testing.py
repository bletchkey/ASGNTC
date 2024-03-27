import sys
import unittest

from config.common import setup_base_directory, setup_logging
from config.constants import *
from config.paths import CONFIG_DIR


from unit_tests.gol_adv_sys.utils.test_scores import TestConfigPredictionAccuracy


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "unit_tests_logging.json")

    suite = unittest.TestSuite()

    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_perfect_match'))
    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_adjacent_bins'))
    suite.addTest(TestConfigPredictionAccuracy('test_accuracy_non_adjacent_bins'))
    suite.addTest(TestConfigPredictionAccuracy('test_input_shape_mismatch'))
    suite.addTest(TestConfigPredictionAccuracy('test_input_dimension_mismatch'))


    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


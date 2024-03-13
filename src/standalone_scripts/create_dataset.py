import numpy as np

import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gol_adv_sys.DatasetManager import DatasetCreator
from gol_adv_sys.DeviceManager import DeviceManager


def main():
    device_manager = DeviceManager()
    dataset = DatasetCreator(device_manager=device_manager)
    dataset.create_dataset()


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.gol_adv_sys.DatasetManager import DatasetCreator
from src.gol_adv_sys.DeviceManager import DeviceManager


def main():
    device_manager = DeviceManager()
    dataset = DatasetCreator(device_manager=device_manager)
    dataset.create_dataset()
    dataset.save_tensors()


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


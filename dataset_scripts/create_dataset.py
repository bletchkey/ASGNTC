from pathlib import Path

from configs.setup import setup_base_directory, setup_logging
from configs.paths import CONFIG_DIR

from src.gol_pred_sys.dataset_manager import DatasetCreator
from src.common.device_manager        import DeviceManager


def main():
    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "main_logging.json")

    device_manager = DeviceManager()
    dataset = DatasetCreator(device_manager=device_manager)
    dataset.create_dataset()
    dataset.save_tensors()


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


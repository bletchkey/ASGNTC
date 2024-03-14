from pathlib import Path
from datetime import datetime

from config.paths import TRAININGS_DIR
from src.gol_adv_sys.utils import constants as constants

class FolderManager:
    """
    Manages folder creation for organizing training sessions, including results, models, and logs.

    This class ensures that all necessary directories exist for storing training outputs,
    segmented by the initiation date and time of the training session.

    Attributes:
        base_folder (str): The base directory for a training sessions.
        results_folder (str): Directory for storing results.
        models_folder (str): Directory for storing model checkpoints.
        logs_folder (str): Directory for storing logs.

    """

    def __init__(self, date: datetime) -> None:

        TRAININGS_DIR.mkdir(parents=True, exist_ok=True)

        self.__base_folder = TRAININGS_DIR / date.strftime("%Y-%m-%d_%H-%M-%S")
        self.__results_dir = self.__base_folder / "results"
        self.__models_dir  = self.__base_folder / "models"
        self.__logs_dir    = self.__base_folder / "logs"

    @property
    def base_folder(self) -> Path:
        """Returns the base folder path, creating it if necessary."""
        if not self.__base_folder.exists():
            self.__base_folder.mkdir(parents=True, exist_ok=True)
        return self.__base_folder

    @property
    def results_folder(self) -> Path:
        """Returns the results folder path, creating it if necessary."""
        if not self.__results_dir.exists():
            self.__results_dir.mkdir(parents=True, exist_ok=True)
        return self.__results_dir

    @property
    def models_folder(self) -> Path:
        """Returns the models folder path, creating it if necessary."""
        if not self.__models_dir.exists():
            self.__models_dir.mkdir(parents=True, exist_ok=True)
        return self.__models_dir

    @property
    def logs_folder(self) -> Path:
        """Returns the logs folder path, creating it if necessary."""
        if not self.__logs_dir.exists():
            self.__logs_dir.mkdir(parents=True, exist_ok=True)
        return self.__logs_dir


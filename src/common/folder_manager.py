from pathlib import Path
from datetime import datetime

from configs.paths import TRAININGS_DIR, TRAININGS_PREDICTOR_DIR, TRAININGS_ADVERSARIAL_DIR
from configs.constants import *

class FolderManager:
    """
    Manages folder creation for organizing training sessions, including results, models, and logs.

    This class ensures that all necessary directories exist for storing training outputs,
    segmented by the initiation date and time of the training session.

    Attributes:
        base_folder (str): The base directory for a training sessions.
        results_folder (str): Directory for storing results.
        checkpoints_folder (str): Directory for storing model checkpoints.
        logs_folder (str): Directory for storing logs.

    """

    def __init__(self, training_type:str, date: datetime) -> None:

        TRAININGS_DIR.mkdir(parents=True, exist_ok=True)

        if training_type == TRAINING_TYPE_PREDICTOR:
            base = TRAININGS_PREDICTOR_DIR / date.strftime("%Y-%m-%d_%H-%M-%S")
        elif training_type == TRAINING_TYPE_ADVERSARIAL:
            base = TRAININGS_ADVERSARIAL_DIR / date.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            raise ValueError(f"Invalid training type: {training_type}")

        self.__base_folder     = base
        self.__results_dir     = self.__base_folder / "results"
        self.__checkpoints_dir = self.__base_folder / "checkpoints"
        self.__logs_dir        = self.__base_folder / "logs"

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
    def checkpoints_folder(self) -> Path:
        """Returns the models folder path, creating it if necessary."""
        if not self.__checkpoints_dir.exists():
            self.__checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return self.__checkpoints_dir

    @property
    def logs_folder(self) -> Path:
        """Returns the logs folder path, creating it if necessary."""
        if not self.__logs_dir.exists():
            self.__logs_dir.mkdir(parents=True, exist_ok=True)
        return self.__logs_dir


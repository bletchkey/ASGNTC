import datetime
import os

from .utils import constants as constants


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

    def __init__(self, date) -> None:
        self.__date_str = date.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(constants.trainings_folder_path):
            os.makedirs(constants.trainings_folder_path, exist_ok=True)

        self.__base_folder    = os.path.join(constants.trainings_folder_path, self.__date_str)
        self.__results_folder = os.path.join(self.__base_folder, constants.results_folder_path)
        self.__models_folder  = os.path.join(self.__base_folder, constants.models_folder_path)
        self.__logs_folder    = os.path.join(self.__base_folder, constants.logs_folder_path)

    @property
    def base_folder(self):
        """Returns the base folder path, creating it if necessary."""
        if not os.path.exists(self.__base_folder):
            self.__base_folder = self.__create_folder(self.__base_folder)

        return self.__base_folder

    @property
    def results_folder(self):
        """Returns the results folder path, creating it if necessary."""
        if not os.path.exists(self.__results_folder):
            self.__results_folder = self.__create_folder(self.__results_folder)

        return self.__results_folder

    @property
    def models_folder(self):
        """Returns the models folder path, creating it if necessary."""
        if not os.path.exists(self.__models_folder):
            self.__models_folder = self.__create_folder(self.__models_folder)

        return self.__models_folder

    @property
    def logs_folder(self):
        """Returns the logs folder path, creating it if necessary."""
        if not os.path.exists(self.__logs_folder):
            self.__logs_folder = self.__create_folder(self.__logs_folder)

        return self.__logs_folder


    def __create_folder(self, path):
        """
        Creates a directory if it does not already exist.

        Parameters:
            path (str): The filesystem path to the directory to create.

        Returns:
            str: The path to the directory, confirming creation or existence.
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path


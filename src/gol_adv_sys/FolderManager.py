import datetime
import os

from .utils import constants as constants


class FolderManager:
    def __init__(self, date) -> None:
        self.__date_str = date.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(constants.trainings_folder_path):
            os.makedirs(constants.trainings_folder_path, exist_ok=True)

        self.__base_folder    = os.path.join(constants.trainings_folder_path, self.__date_str)
        self.__results_folder = os.path.join(self.__base_folder, constants.results_folder_path)
        self.__models_folder  = os.path.join(self.__base_folder, constants.models_folder_path)
        self.__logs_folder    = os.path.join(self.__base_folder, constants.logs_folder_path)

        self.__data_folder    = constants.fixed_dataset_path

    @property
    def base_folder(self):
        if not os.path.exists(self.__base_folder):
            self.__base_folder = self.__create_folder(self.__base_folder)

        return self.__base_folder

    @property
    def results_folder(self):
        if not os.path.exists(self.__results_folder):
            self.__results_folder = self.__create_folder(self.__results_folder)

        return self.__results_folder

    @property
    def models_folder(self):
        if not os.path.exists(self.__models_folder):
            self.__models_folder = self.__create_folder(self.__models_folder)

        return self.__models_folder

    @property
    def logs_folder(self):
        if not os.path.exists(self.__logs_folder):
            self.__logs_folder = self.__create_folder(self.__logs_folder)

        return self.__logs_folder

    @property
    def data_folder(self):
        if not os.path.exists(self.__data_folder):
            self.__data_folder = self.__create_folder(self.__data_folder)

        return self.__data_folder


    def __create_folder(self, path):

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path


import datetime
import os

from . import constants as constants


class training_folders:
    def __init__(self, date):
        self.__date_str = date.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(constants.trainings_folder_path):
            os.makedirs(constants.trainings_folder_path, exist_ok=True)

        self.__base_path    = os.path.join(constants.trainings_folder_path, self.__date_str)
        self.__results_path = os.path.join(self.__base_path, constants.results_folder_path)
        self.__models_path  = os.path.join(self.__base_path, constants.models_folder_path)
        self.__logs_path    = os.path.join(self.__base_path, constants.logs_folder_path)

    @property
    def results_path(self):
        if not os.path.exists(self.__results_path):
            self.__results_path = self.__create_folder(self.__results_path)

        return self.__results_path

    @property
    def models_path(self):
        if not os.path.exists(self.__models_path):
            self.__models_path = self.__create_folder(self.__models_path)

        return self.__models_path

    @property
    def logs_path(self):
        if not os.path.exists(self.__logs_path):
            self.__logs_path = self.__create_folder(self.__logs_path)

        return self.__logs_path


    def __create_folder(self, path):

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path


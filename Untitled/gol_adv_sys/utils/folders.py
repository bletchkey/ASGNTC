import datetime
import os

from . import constants as constants


class training_folders:
    def __init__(self, date):
        self.__date_str = date.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(constants.results_folder_path):
            os.makedirs(constants.results_folder_path, exist_ok=True)

        if not os.path.exists(constants.models_folder_path):
            os.makedirs(constants.models_folder_path, exist_ok=True)

        self.__results_path = None
        self.__models_path  = None

    @property
    def results_path(self):
        if self.__results_path is None:
            self.__results_path = self.__create_folder(constants.results_folder_path, self.__date_str)
        return self.__results_path

    @property
    def models_path(self):
        if self.__models_path is None:
            self.__models_path = self.__create_folder(constants.models_folder_path, self.__date_str)
        return self.__models_path

    def __create_folder(self, base_path, date_str):

        #create new folder
        new_folder_path = os.path.join(base_path, date_str)
        os.makedirs(new_folder_path, exist_ok=True)
        return new_folder_path


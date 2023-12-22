import datetime
import os

from . import constants as constants


class training_folders:
    def __init__(self, date):
        self.__date_str = date.strftime("%Y-%m-%d_%H-%M-%S")
        self.__results_path = self.__create_folder(constants.results_folder_path, self.__date_str)
        self.__models_path  = self.__create_folder(constants.models_folder_path, self.__date_str)

    @property
    def results_path(self):
        return self.__results_path

    @property
    def models_path(self):
        return self.__models_path

    def __create_folder(self, base_path, date_str):

        #check base path exists and create if not
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        #create new folder
        new_folder_path = os.path.join(base_path, date_str)
        os.makedirs(new_folder_path, exist_ok=True)
        return new_folder_path


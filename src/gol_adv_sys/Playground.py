import typing
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.gol_adv_sys.DeviceManager import DeviceManager
from src.gol_adv_sys.utils.simulation_functions import simulate_config
from src.gol_adv_sys.utils import constants as constants

from src.gol_adv_sys.DatasetManager import FixedDataset

from config.paths import DATASET_DIR

class Playground():

    def __init__(self):
        self.__device_manager = DeviceManager()
        self.__dataset = {"train_data": None, "train_meta": None}

    @property
    def topology(self) -> int:
        return self.__topology

    @property
    def train_dataset(self) -> FixedDataset:
        return self.__dataset["train"]

    @property
    def val_dataset(self) -> FixedDataset:
        return self.__dataset["val"]

    @property
    def test_dataset(self) -> FixedDataset:
        return self.__dataset["test"]

    @property
    def train_dataloader(self) -> DataLoader:
        return self.__dataloader["train"]

    @property
    def val_dataloader(self) -> DataLoader:
        return self.__dataloader["val"]

    @property
    def test_dataloader(self) -> DataLoader:
        return self.__dataloader["test"]


    def simulate(self, config: torch.Tensor, steps: int, calc_final: bool=True) -> torch.Tensor:

        config = config.to(self.__device_manager.default_device)

        final, metrics, n_cells_init, n_cells_final = simulate_config(config, constants.TOPOLOGY_TYPE["toroidal"], steps=steps,
                                                        calculate_final_config=calc_final, device=self.__device_manager.default_device)


        results = {
            "period": final["period"],
            "antiperiod": final["antiperiod"],
            "n_cells_init": n_cells_init,
            "n_cells_final": n_cells_final,
            "final_config": final["config"],
            "easy_metric": metrics["easy"],
            "medium_metric": metrics["medium"],
            "hard_metric": metrics["hard"]
        }

        return results


    def get_config_id(self, id: int) -> typing.Tuple[torch.Tensor, dict]:

        if self.__dataset["train_data"] is None:
            self.__load_train_dataset()
        if self.__dataset["train_meta"] is None:
            self.__load_train_metadata()

        for data, meta in zip(self.__dataset["train_data"], self.__dataset["train_meta"]):
            if meta["id"] == id:
                return self.__create_data(data, meta)


        raise ValueError(f"Could not find the data for id {id}")


    def plot_antiperiods(self) -> None:

        if self.__dataset["train_meta"] is None:
            self.__load_train_metadata()

        pairs = []
        for meta in self.__dataset["train_meta"]:
            id = meta["id"]
            antiperiod = meta["antiperiod"]
            pairs.append((id, antiperiod))

        pairs.sort(key=lambda x: x[1])

        ids, antiperiods = zip(*pairs)

        plt.figure(figsize=(10, 6))
        plt.bar(ids, antiperiods)
        plt.xlabel("ID")
        plt.ylabel("Antiperiod")
        plt.title("Antiperiods")
        plt.grid(True)
        plt.savefig("antiperiods_bar_graph.png", dpi=300)
        plt.close()


    def __load_train_dataset(self):

        train_data_path = DATASET_DIR / f"{constants.dataset_name}_train.pt"
        self.__dataset["train_data"] = FixedDataset(train_data_path)


    def __load_train_metadata(self):

        train_meta_path = DATASET_DIR / f"{constants.dataset_name}_metadata_train.pt"
        self.__dataset["train_meta"] = torch.load(train_meta_path)


    def __create_data(self, data: torch.Tensor, metadata: dict) -> typing.Tuple[torch.Tensor, dict]:

        informations = {
            "id"            : metadata["id"],
            "n_cells_init"  : metadata["n_cells_init"],
            "n_cells_final" : metadata["n_cells_final"],
            "period"        : metadata["period"],
            "antiperiod"    : metadata["antiperiod"],
            "initial_config": data[0, :, :, :],
            "final_config"  : data[1, :, :, :],
            "easy_metric"   : data[2, :, :, :],
            "medium_metric" : data[3, :, :, :],
            "hard_metric"   : data[4, :, :, :]
        }

        return informations


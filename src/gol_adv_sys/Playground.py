import typing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import torch
from torch.utils.data import DataLoader

from src.gol_adv_sys.utils.simulation_functions import simulate_config
from config.constants import *

from src.gol_adv_sys.DatasetManager import FixedDataset
from src.gol_adv_sys.DeviceManager import DeviceManager

from config.paths import DATASET_DIR, TRAINED_MODELS_DIR

class Playground():

    def __init__(self):
        self.__device_manager = DeviceManager()
        self.__dataset = {"train_data": None, "train_meta": None}
        self.__predictor = {"model": None, "name": None}

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


    def simulate(self, config: torch.Tensor, steps: int) -> torch.Tensor:

        config = config.to(self.__device_manager.default_device)
        final, metrics, n_cells_init, n_cells_final = simulate_config(config, TOPOLOGY_TOROIDAL, steps=steps,
                                                        calculate_final_config=True, device=self.__device_manager.default_device)


        results = {
            "period": final["period"],
            "transient_phase": final["transient_phase"],
            "n_cells_init": n_cells_init,
            "n_cells_final": n_cells_final,
            "final_config": final["config"],
            "easy": metrics["easy"]["config"],
            "medium": metrics["medium"]["config"],
            "hard": metrics["hard"]["config"],
            "stable": metrics["stable"]["config"]

        }

        return results


    def get_record_from_id(self, id: int) -> typing.Tuple[torch.Tensor, dict]:

        if self.__dataset["train_data"] is None:
            self.__load_train_dataset()
        if self.__dataset["train_meta"] is None:
            self.__load_train_metadata()

        for data, meta in zip(self.__dataset["train_data"], self.__dataset["train_meta"]):
            if meta["id"] == id:
                return self.__create_record_dict(data, meta)


        raise ValueError(f"Could not find the data for id {id}")


    def load_predictor(self, name: str) -> None:

        model_path = TRAINED_MODELS_DIR / name
        if not model_path.exists():
            raise ValueError(f"Model {name} does not exist")

        self.__predictor["model"] = torch.load(model_path)
        self.__predictor["name"] = name


    def predict(self, config: torch.Tensor) -> torch.Tensor:

        if self.__predictor is None:
            raise ValueError("Predictor model has not been loaded")

        self.__predictor["model"].eval()

        with torch.no_grad():
            return self.__predictor["model"](config)


    def plot_record(self, record: dict) -> None:
        fig = plt.figure(figsize=(24, 12))

        gs = GridSpec(2, 6, figure=fig, height_ratios=[6, 1], hspace=0.1, wspace=0.1)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}

        titles = ["Initial Configuration", "Final Configuration", "Easy Metric",
                  "Medium Metric", "Hard Metric", "Stable Metric"]

        # Image plots in the first row
        for i, config in enumerate(["initial_config", "final_config", "easy", "medium", "hard", "stable"]):
            if i == 0:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Initial - {record['n_cells_init']} cells", fontsize=16)
                ax.axis('off')
                continue
            if i == 1:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Final - {record['n_cells_final']} cells", fontsize=16)
                ax.axis('off')
                continue
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(record[f"{config}"]["config"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
            ax.set_title(f"{titles[i]}", fontsize=16)
            ax.axis('off')

        # Text plot for metrics in the first row
        configs_types_list = [CONFIG_INITIAL, CONFIG_FINAL,
                              CONFIG_METRIC_EASY, CONFIG_METRIC_MEDIUM,
                              CONFIG_METRIC_HARD, CONFIG_METRIC_STABLE]
        for i, config in enumerate(configs_types_list):
            ax = fig.add_subplot(gs[1, i])
            if config == CONFIG_INITIAL:
                text_str = f"ID: {record['id']}"
                ax.text(0.1, 0, text_str, ha="left", va="center", fontsize=14, wrap=True)
                ax.axis('off')
                continue

            if config == CONFIG_FINAL:
                text_str = f"Transient phase: {record['transient_phase']}\nPeriod: {record['period']}"
                ax.text(0.1, 0, text_str, ha="left", va="center", fontsize=14, wrap=True)
                ax.axis('off')
                continue

            text_str = record[f"{config}"]
            ax.text(0.1, 0, f"Min: {text_str['minimum']}\nMax: {text_str['maximum']}\n"
                                f"Q1: {text_str['q1']}\nQ2: {text_str['q2']}\nQ3: {text_str['q3']}",
                    ha="left", va="center", fontsize=14, wrap=True)
            ax.axis('off')


        # Adjust layout for padding and spacing

        plt.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.1, wspace=0.1, hspace=0)

        # Save and close
        plt.savefig(f"record_{record['id']}.png", dpi = 600, bbox_inches='tight')
        plt.close(fig)


    def __load_train_dataset(self):

        train_data_path = DATASET_DIR / f"{DATASET_NAME}_train.pt"
        self.__dataset["train_data"] = FixedDataset(train_data_path)


    def __load_train_metadata(self):

        train_meta_path = DATASET_DIR / f"{DATASET_NAME}_metadata_train.pt"
        self.__dataset["train_meta"] = torch.load(train_meta_path)


    def __create_record_dict(self, data: torch.Tensor, metadata: dict) -> typing.Tuple[torch.Tensor, dict]:

        informations = {
            "id"              : metadata["id"],
            "n_cells_init"    : metadata["n_cells_init"],
            "n_cells_final"   : metadata["n_cells_final"],
            "period"          : metadata["period"],
            "transient_phase" : metadata["transient_phase"],
            "initial_config"  : data[0, :, :, :],
            "final_config"    : data[1, :, :, :],
            "easy"     : {
                "config"  : data[2, :, :, :],
                "minimum" : metadata["easy_minimum"],
                "maximum" : metadata["easy_maximum"],
                "q1"      : metadata["easy_q1"],
                "q2"      : metadata["easy_q2"],
                "q3"      : metadata["easy_q3"],
            },
            "medium"  : {
                "config"  : data[3, :, :, :],
                "minimum" : metadata["medium_minimum"],
                "maximum" : metadata["medium_maximum"],
                "q1"      : metadata["medium_q1"],
                "q2"      : metadata["medium_q2"],
                "q3"      : metadata["medium_q3"],
            },
            "hard"    : {
                "config"  : data[4, :, :, :],
                "minimum" : metadata["hard_minimum"],
                "maximum" : metadata["hard_maximum"],
                "q1"      : metadata["hard_q1"],
                "q2"      : metadata["hard_q2"],
                "q3"      : metadata["hard_q3"],
            },
            "stable"  : {
                "config"  : data[5, :, :, :],
                "minimum" : metadata["stable_minimum"],
                "maximum" : metadata["stable_maximum"],
                "q1"      : metadata["stable_q1"],
                "q2"      : metadata["stable_q2"],
                "q3"      : metadata["stable_q3"],
            }
        }

        return informations


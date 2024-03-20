import typing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import torch
from torch.utils.data import DataLoader

from src.gol_adv_sys.utils.simulation_functions import simulate_config
from src.gol_adv_sys.utils import constants as constants

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
        fig = plt.figure(figsize=(30, 10))
        gs = GridSpec(2, 5, figure=fig)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}

        axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
        axs.append(fig.add_subplot(gs[1, :]))

        axs[0].imshow(record["initial_config"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
        axs[0].set_title(f"Initial Configuration - {record['n_cells_init']} cells")
        axs[1].imshow(record["final_config"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
        axs[1].set_title(f"Final Configuration - {record['n_cells_final']} cells")
        axs[2].imshow(record["easy_metric"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
        axs[2].set_title("Easy Metric")
        axs[3].imshow(record["medium_metric"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
        axs[3].set_title("Medium Metric")
        axs[4].imshow(record["hard_metric"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
        axs[4].set_title("Hard Metric")

        text_str = f"ID: {record['id']}\nPeriod: {record['period']}\nAntiperiod: {record['antiperiod']}"
        axs[5].text(0.5, 0.5, text_str, ha="center", va="center", fontsize=30, wrap=True)
        axs[5].axis("off")

        plt.tight_layout(pad=1.0)

        plt.savefig(f"record_{record['id']}.png")
        plt.close(fig)


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


    def __create_record_dict(self, data: torch.Tensor, metadata: dict) -> typing.Tuple[torch.Tensor, dict]:

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


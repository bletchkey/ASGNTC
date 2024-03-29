import typing
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from src.common.utils.simulation_functions import simulate_config
from configs.constants import *
from configs.paths import DATASET_DIR, TRAINED_MODELS_DIR

from src.gol_pred_sys.dataset_manager import FixedDataset
from src.common.device_manager import DeviceManager


class Playground():

    def __init__(self):
        self.__device_manager = DeviceManager()
        self.__dataset = {TRAIN: None, TRAIN_METADATA: None}
        self.__predictor = {"model": None, "name": None}

    @property
    def topology(self) -> int:
        return self.__topology

    @property
    def train_data(self) -> FixedDataset:
        return self.__dataset[TRAIN]

    @property
    def train_meta(self) -> dict:
        return self.__dataset[TRAIN_METADATA]


    def simulate(self, config: torch.Tensor, steps: int) -> torch.Tensor:

        config = config.to(self.__device_manager.default_device)
        sim_results = simulate_config(config, TOPOLOGY_TOROIDAL, steps=steps,
                                  device=self.__device_manager.default_device)

        final           = sim_results["final"]
        n_cells_initial = sim_results["n_cells_initial"]
        n_cells_final   = sim_results["n_cells_final"]
        metrics         = sim_results["all_metrics"]

        results = {
            META_PERIOD: final[META_PERIOD],
            META_TRANSIENT_PHASE: final[META_TRANSIENT_PHASE],
            META_N_CELLS_INITIAL : n_cells_initial,
            META_N_CELLS_FINAL: n_cells_final,
            CONFIG_FINAL: final[CONFIG_FINAL]["config"],
            CONFIG_METRIC_EASY: metrics[CONFIG_METRIC_EASY]["config"],
            CONFIG_METRIC_MEDIUM: metrics[CONFIG_METRIC_MEDIUM]["config"],
            CONFIG_METRIC_HARD: metrics[CONFIG_METRIC_HARD]["config"],
            CONFIG_METRIC_STABLE: metrics[CONFIG_METRIC_STABLE]["config"]
        }

        return results


    def get_record_from_id(self, id: int) -> typing.Tuple[torch.Tensor, dict]:

        if self.__dataset[TRAIN] is None:
            self.__load_train_dataset()
        if self.__dataset[TRAIN_METADATA] is None:
            self.__load_train_metadata()

        for data, meta in zip(self.__dataset[TRAIN], self.__dataset[TRAIN_METADATA]):
            if meta[META_ID] == id:
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
                ax.set_title(f"Initial - {record[META_N_CELLS_INITIAL ]} cells", fontsize=16)
                ax.axis('off')
                continue
            if i == 1:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Final - {record[META_N_CELLS_FINAL]} cells", fontsize=16)
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
                text_str = f"ID: {record[META_ID]}"
                ax.text(0.1, 0, text_str, ha="left", va="center", fontsize=14, wrap=True)
                ax.axis('off')
                continue

            if config == CONFIG_FINAL:
                text_str = f"Transient phase: {record[META_TRANSIENT_PHASE]}\nPeriod: {record[META_PERIOD]}"
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
        self.__dataset[TRAIN_METADATA] = torch.load(train_meta_path)


    def __create_record_dict(self, data: torch.Tensor, metadata: dict) -> typing.Tuple[torch.Tensor, dict]:

        informations = {
            "id"              : metadata[META_ID],
            "n_cells_init"    : metadata[META_N_CELLS_INITIAL],
            "n_cells_final"   : metadata[META_N_CELLS_FINAL],
            "period"          : metadata[META_PERIOD],
            "transient_phase" : metadata[META_TRANSIENT_PHASE],
            "initial_config"  : data[0, :, :, :],
            "final_config"    : data[1, :, :, :],
            "easy"     : {
                "config"  : data[2, :, :, :],
                "minimum" : metadata[META_EASY_MIN],
                "maximum" : metadata[META_EASY_MAX],
                "q1"      : metadata[META_EASY_Q1],
                "q2"      : metadata[META_EASY_Q2],
                "q3"      : metadata[META_EASY_Q3],
            },
            "medium"  : {
                "config"  : data[3, :, :, :],
                "minimum" : metadata[META_MEDIUM_MIN],
                "maximum" : metadata[META_MEDIUM_MAX],
                "q1"      : metadata[META_MEDIUM_Q1],
                "q2"      : metadata[META_MEDIUM_Q2],
                "q3"      : metadata[META_MEDIUM_Q3],
            },
            "hard"    : {
                "config"  : data[4, :, :, :],
                "minimum" : metadata[META_HARD_MIN],
                "maximum" : metadata[META_HARD_MAX],
                "q1"      : metadata[META_HARD_Q1],
                "q2"      : metadata[META_HARD_Q2],
                "q3"      : metadata[META_HARD_Q3],
            },
            "stable"  : {
                "config"  : data[5, :, :, :],
                "minimum" : metadata[META_STABLE_MIN],
                "maximum" : metadata[META_STABLE_MAX],
                "q1"      : metadata[META_STABLE_Q1],
                "q2"      : metadata[META_STABLE_Q2],
                "q3"      : metadata[META_STABLE_Q3],
            }
        }

        return informations


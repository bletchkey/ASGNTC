import typing
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from src.common.utils.simulation_functions import simulate_config, basic_simulation_config
from configs.constants import *
from configs.paths import DATASET_DIR, TRAINED_MODELS_DIR

from src.gol_pred_sys.dataset_manager import FixedDataset
from src.common.device_manager        import DeviceManager

from src.common.generator import Generator_Gambler


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


    def simulate(self, config: torch.Tensor, steps: int, topology: topology) -> torch.Tensor:

        initial = config
        config = config.to(self.__device_manager.default_device)
        sim_results = simulate_config(config, topology, steps=steps,
                                      device=self.__device_manager.default_device)

        period            = sim_results["period"].detach().cpu().numpy()
        transient_phase   = sim_results["transient_phase"].detach().cpu().numpy()
        simulated         = sim_results["simulated"]
        final             = sim_results["final"]
        n_cells_initial   = sim_results["n_cells_initial"].detach().cpu().numpy()
        n_cells_simulated = sim_results["n_cells_simulated"].detach().cpu().numpy()
        n_cells_final     = sim_results["n_cells_final"].detach().cpu().numpy()
        metrics           = sim_results["all_metrics"]

        results = {
            "period": period,
            "transient_phase": transient_phase,
            "n_cells_initial" : n_cells_initial,
            "n_cells_simulated": n_cells_simulated,
            "n_cells_final": n_cells_final,
            "initial": initial,
            "simulated": simulated,
            "final": final,
            "steps": steps,
            "easy": metrics[CONFIG_TARGET_EASY]["config"],
            "medium": metrics[CONFIG_TARGET_MEDIUM]["config"],
            "hard": metrics[CONFIG_TARGET_HARD]["config"],
            "stable": metrics[CONFIG_TARGET_STABLE]["config"]
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


    def plot_record_db(self, record: dict) -> None:
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
                              CONFIG_TARGET_EASY, CONFIG_TARGET_MEDIUM,
                              CONFIG_TARGET_HARD, CONFIG_TARGET_STABLE]
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


    def plot_record_sim(self, record: dict) -> None:
        fig = plt.figure(figsize=(24, 12))

        gs = GridSpec(2, 7, figure=fig, height_ratios=[7, 1], hspace=0.1, wspace=0.1)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}

        titles = ["Initial Configuration", "Simulated Configuration","Final Configuration", "Easy Metric",
                  "Medium Metric", "Hard Metric", "Stable Metric"]

        # Image plots in the first row
        for i, config in enumerate(["initial", "simulated", "final", "easy", "medium", "hard", "stable"]):
            if i == 0:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Initial - {record['n_cells_initial']} cells", fontsize=16)
                ax.axis('off')
                continue
            if i == 1:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Simulated - {record['n_cells_simulated']} cells", fontsize=16)
                ax.axis('off')
                continue
            if i == 2:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                ax.set_title(f"Final - {record['n_cells_final']} cells", fontsize=16)
                ax.axis('off')
                continue

            ax = fig.add_subplot(gs[0, i])
            ax.imshow(record[f"{config}"].detach().cpu().numpy().squeeze(), **imshow_kwargs)
            ax.set_title(f"{titles[i]}", fontsize=16)

            ax.axis('off')

        # Text plot for metrics in the first row
        configs_types_list = ["initial", "simulated", "final",
                              "easy", "medium", "hard", "stable"]

        for i, config in enumerate(configs_types_list):
            ax = fig.add_subplot(gs[1, i])

            if config == "simulated":
                text_str = f"Steps: {record['steps']}"
                ax.text(0.1, 0, text_str, ha="left", va="center", fontsize=14, wrap=True)
                ax.axis('off')
                continue

            if config == "final":
                text_str = f"Transient phase: {record['transient_phase']}\nPeriod: {record['period']}"
                ax.text(0.1, 0, text_str, ha="left", va="center", fontsize=14, wrap=True)
                ax.axis('off')
                continue

            ax.axis('off')

        # Adjust layout for padding and spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.1, wspace=0.1, hspace=0)

        # Save and close
        plt.savefig(f"simulation_results.png", dpi = 600, bbox_inches='tight')
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


    def ulam_spiral(self, size: int) -> torch.Tensor:

        spiral = torch.zeros((size, size), dtype=torch.float32, device=self.__device_manager.default_device)

        # Define the starting point and the initial direction
        x, y = size // 2, size // 2
        spiral[x, y] = 0

        # Define movement directions (right, up, left, down)
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        direction_index = 0  # Start with moving right
        num = 1  # Start with the first number
        steps = 1  # Steps to take in the current direction
        num_changes = 0  # Count when we change direction

        while 0 <= x < size and 0 <= y < size:
            for _ in range(2):  # Change direction twice after completing two sides of a square
                for _ in range(steps):
                    if 0 <= x < size and 0 <= y < size:
                        # Check if num is prime
                        if num > 1:
                            is_prime = True
                            for i in range(2, int(torch.sqrt(torch.tensor(num)).item()) + 1):
                                if num % i == 0:
                                    is_prime = False
                                    break
                            spiral[x, y] = 1 if is_prime else 0
                        num += 1
                    # Move in the current direction
                    dx, dy = directions[direction_index]
                    x += dx
                    y += dy
                direction_index = (direction_index + 1) % 4  # Change direction
                num_changes += 1
                if num_changes % 2 == 0:
                    steps += 1  # Increase steps after completing a layer

        return spiral


    def generate_gambler(self, batch_size:int) -> torch.Tensor:

        generator = Generator_Gambler().to(self.__device_manager.default_device)

        generator.eval()

        config = torch.zeros(batch_size, GRID_NUM_CHANNELS, GRID_SIZE, GRID_SIZE,
                             device=self.__device_manager.default_device)
        config[:, :, GRID_SIZE // 2, GRID_SIZE // 2] = 1

        generated_config, probabilities = generator(config)

        return generated_config, probabilities


    def gol_basic_simulation(self, config: torch.Tensor, steps: int, topology: topology) -> torch.Tensor:

        config = config.to(self.__device_manager.default_device)
        configs = basic_simulation_config(config, steps=steps, topology=topology,
                                          device=self.__device_manager.default_device)

        return configs


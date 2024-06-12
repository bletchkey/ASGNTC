import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.utils.helpers import export_figures_to_pdf

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR, OUTPUTS_DIR

from src.common.playground            import Playground
from src.gol_pred_sys.dataset_manager import DatasetManager


def get_glider():
    config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    config[0, 0, 15, 15] = 1
    config[0, 0, 15, 16] = 1
    config[0, 0, 16, 15] = 1
    config[0, 0, 16, 14] = 1
    config[0, 0, 14, 14] = 1

    return config


def plot_toroidal_focus(configs, title):
    n = 10
    state = configs[n]
    state = state.squeeze().detach().cpu().numpy()
    wrap_matrix = state.copy()
    wrap_matrix /= 2
    state_wrapped = np.block([
    [wrap_matrix, wrap_matrix, wrap_matrix],
    [wrap_matrix, state, wrap_matrix],
    [wrap_matrix, wrap_matrix, wrap_matrix]
    ])

    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    axs.imshow(state_wrapped, cmap='gray', vmin=0, vmax=1)
    axs.set_frame_on(True)
    axs.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle(f"Step {n}", fontsize=32, y=0.985)
    plt.tight_layout()

    pdf_path = OUTPUTS_DIR / f"{title}.pdf"
    export_figures_to_pdf(pdf_path, fig)


def plot_configs(configs, title):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(16):
        ax = axs[i//4, i%4]
        ax.imshow(configs[i].detach().cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f"Step {i}")
        ax.axis('off')

    for spine in ax.spines.values():
          spine.set_edgecolor('black')
          spine.set_linewidth(1)


    plt.tight_layout()

    pdf_path = OUTPUTS_DIR / f"{title}.pdf"
    export_figures_to_pdf(pdf_path, fig)


def simulate_heart():

    init_config = torch.zeros(1, 1, 15, 15, dtype=torch.float32)

    # Heart
    init_config[0, 0, 6, 7]  = 1
    init_config[0, 0, 10, 7] = 1
    init_config[0, 0, 9, 6]  = 1
    init_config[0, 0, 9, 8]  = 1
    init_config[0, 0, 8, 5]  = 1
    init_config[0, 0, 8, 9]  = 1
    init_config[0, 0, 7, 4]  = 1
    init_config[0, 0, 6, 4]  = 1
    init_config[0, 0, 7, 10] = 1
    init_config[0, 0, 6, 10] = 1
    init_config[0, 0, 5, 5]  = 1
    init_config[0, 0, 5, 6]  = 1
    init_config[0, 0, 5, 8]  = 1
    init_config[0, 0, 5, 9]  = 1


    pg = Playground()

    results_toroidal = pg.gol_basic_simulation(init_config, steps=16, topology=TOPOLOGY_TOROIDAL)
    results_flat     = pg.gol_basic_simulation(init_config, steps=16, topology=TOPOLOGY_FLAT)

    # plot 16 grids in an image
    plot_configs(results_toroidal, "toroidal")
    plot_toroidal_focus(results_toroidal, "toroidal_focus")
    plot_configs(results_flat, "flat")


def simulate_configs():

    pg = Playground()

    config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    config[0, 0, 15, 15] = 1
    config[0, 0, 15, 16] = 1
    config[0, 0, 15, 17] = 1

    results = pg.simulate(config, steps=100, topology=TOPOLOGY_TOROIDAL)
    pg.plot_record_sim(results)

    config  = get_glider()
    results = pg.simulate(config, steps=200, topology=TOPOLOGY_TOROIDAL)
    pg.plot_record_sim(results)


def plot_record_structure():

    pg = Playground()

    initial_cells = [100, 200, 300, 400, 500]

    for cells in initial_cells:
        data = pg.get_record_from_id(2 * DATASET_BATCH_SIZE * cells)
        pg.plot_record_db(data)


def plot_targets():
    pg = Playground()

    data = pg.get_record_from_id(2 * DATASET_BATCH_SIZE * 320)
    pg.plot_targets(data)


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_playground.json")

    # plot_record_structure()
    plot_targets()
    # simulate_configs()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


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


def test_r_pentomino_5x5_toroidal():
    pg = Playground()

    grid_size = [4, 5, 8, 16, 32, 64, 128, 256, 512]
    for size in grid_size:

        init_config = torch.zeros(1, 1, size, size, dtype=torch.float32)
        #center r-pentomino
        init_config[0, 0, size//2, size//2] = 1
        init_config[0, 0, size//2+1, size//2] = 1
        init_config[0, 0, size//2-1, size//2] = 1
        init_config[0, 0, size//2, size//2-1] = 1
        init_config[0, 0, size//2-1, size//2+1] = 1

        # # print every number in init_config.squeeze().detach().cpu().numpy()
        # for i in range(size):
        #     for j in range(size):
        #         print(int(init_config.squeeze().detach().cpu().numpy()[i][j]), end=" ")

        #     print()

        results = pg.simulate(init_config, steps=3000, topology=TOPOLOGY_FLAT)

        period            = results['period'].item()
        transient_phase   = results['transient_phase'].item()
        n_cells_initial   = results['n_cells_initial'].item()
        n_cells_final     = results['n_cells_final'].item()
        n_cells_simulated = results['n_cells_simulated'].item()
        steps             = results['steps']

        print(f"Grid size: {size}")
        print(f"Period: {period}")
        print(f"Transient phase: {transient_phase}")
        print(f"Number of initial cells: {n_cells_initial}")
        print(f"Number of final cells: {n_cells_final}")
        print(f"Number of simulated cells: {n_cells_simulated}")
        print(f"Number of steps: {steps}")

        print(f"---------------------------------")


def period_analysis():

    pg = Playground()

    grid_size = 32

    n_configs = 256
    initial_configs = torch.zeros(n_configs, 1, grid_size, grid_size)

    # Random initial configurations
    for i in range(n_configs):
        initial_configs[i, 0, :, :] = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.float32)

    dataloader = torch.utils.data.DataLoader(initial_configs, batch_size=64, shuffle=True)

    for topology in [TOPOLOGY_TOROIDAL, TOPOLOGY_FLAT]:
        print(f"Topology: {topology}\n")
        for i, batch in enumerate(dataloader):
            results = pg.simulate(batch, steps=1000, topology=topology)

            period            = results['period']
            transient_phase   = results['transient_phase']

            print(f"Batch {i}")
            print(f"Period: {period}")
            print(f"Tansient phase: {transient_phase}")

            print(f"\nAverages")
            print(f"Period: {period.mean()}")
            print(f"Tansient phase: {transient_phase.mean()}")


def get_id_from_period(period):

    dm = DatasetManager()

    ids = []
    for i in [TRAIN, VALIDATION, TEST]:
        m_i = dm.get_metadata(i)

        for j in m_i:
            if j[META_PERIOD] == period:
                ids.append(j[META_ID])


    return ids

def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_playground.json")

    # plot_record_structure()
    # plot_targets()
    # simulate_configs()
    # test_r_pentomino_5x5_toroidal()

    #period_analysis()

    # res = get_id_from_period(64)

    # print(res)

    pg = Playground()

    id = 95511
    # res = pg.get_record_from_id(id)

    # pg.plot_record_sim(res)

    dm = DatasetManager()

    combined_dataset = dm.get_combined_dataset()

    for data, metadata in combined_dataset:
            if metadata[META_ID] == id:
                config_initial = data[0]
                break

    print(config_initial)

    # results = pg.simulate(config_initial, steps=2000, topology=TOPOLOGY_TOROIDAL)
    # pg.plot_record_sim(results)

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


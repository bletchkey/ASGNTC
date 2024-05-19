import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.setup import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths import CONFIG_DIR

from src.common.playground          import Playground


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
    plt.savefig(f"{title}.png")


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
    plt.savefig(f"{title}.png")


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


def playground():

    # init_config = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    # n_init_cells = 512
    # indices = torch.randperm(32*32)[:n_init_cells]
    # rows, cols = indices // 32, indices % 32
    # init_config[0, 0, rows, cols] = 1

    # pg = Playground()

    # results = pg.simulate(init_config, steps=10)

    # print(f"Stable min: {results[CONFIG_TARGET_STABLE].min()}")
    # print(f"Stable max: {results[CONFIG_TARGET_STABLE].max()}")

    # data = pg.get_record_from_id(200000)
    # pg.plot_record_db(data)

    pg = Playground()

    # inital_config = pg.ulam_spiral(GRID_SIZE)
    # inital_config = inital_config.unsqueeze(0).unsqueeze(0)
    # results       = pg.simulate(inital_config, steps=1, topology=TOPOLOGY_TOROIDAL)
    # pg.plot_record_sim(results)

    config, prob = pg.generate_gambler(4)

    for c in config:
        print(c)
        print(torch.argmax(c.view(-1)))
        print(torch.sum(c.view(-1)))

    print(f"Prob: {prob}")


    config, prob = pg.generate_gambler(BATCH_SIZE)
    results      = pg.simulate(config, steps=20, topology=TOPOLOGY_TOROIDAL)
    pg.plot_record_sim(results)


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_playground.json")

    simulate_heart()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


import sys

import torch
from gol_adv_sys.utils import constants as constants
from gol_adv_sys.training import Training
from gol_adv_sys.utils import helper_functions as hf

def test():
    plot_data = {
        "initial_conf": None,
        "simulated_conf": None,
        "predicted_metric": None,
        "simulated_metric": None,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 300
    grid = torch.zeros((1, 1, 32, 32), dtype=torch.float32, device=device)

    # create glider at position (1, 1)
    grid[0, 0, 1, 1] = 1.
    grid[0, 0, 2, 2] = 1.
    grid[0, 0, 2, 3] = 1.
    grid[0, 0, 3, 1] = 1.
    grid[0, 0, 3, 2] = 1.

    # create glider at position (25, 1)

    grid[0, 0, 25, 1] = 1.
    grid[0, 0, 26, 2] = 1.
    grid[0, 0, 26, 3] = 1.
    grid[0, 0, 27, 1] = 1.
    grid[0, 0, 27, 2] = 1.


    plot_data["initial_conf"] = grid[0]
    grid, metric = hf.simulate_grid(grid, constants.TOPOLOGY["toroidal"], steps, device)
    plot_data["simulated_conf"] = grid[0]
    plot_data["predicted_metric"] = torch.ones_like(metric[0])
    plot_data["simulated_metric"] = metric[0]


    hf.save_progress_plot(plot_data, 0)

def main():

    train = Training()
    train.run()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


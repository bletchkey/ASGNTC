import numpy as np

import sys
import os
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gol_adv_sys.DatasetManager import DatasetCreator
from gol_adv_sys.DeviceManager import DeviceManager

import gol_adv_sys.utils.constants as constants


def check_fixed_dataset_distribution(device, dataset_path):
    n = constants.grid_size ** 2
    bins = torch.zeros(n+1, dtype=torch.int64, device=device)
    img_path = dataset_path + "_distribution.png"

    # Load the dataset
    dataset = torch.load(dataset_path, map_location=device)

    for config in dataset:
        config_flat = config[0].view(-1)
        living_cells = int(torch.sum(config_flat).item())

        if living_cells <= n:
            bins[living_cells] += 1
        else:
            print(f"Warning: Living cells count {living_cells} exceeds expected range.")

    # Move the bins tensor back to CPU for plotting
    bins_cpu = bins.to('cpu').numpy()
    max_value = bins_cpu.max()

    # Plotting
    plt.bar(range(n+1), bins_cpu)
    plt.ylim(0, max_value + max_value * 0.1)  # Add 10% headroom above the max value
    plt.draw()  # Force redraw with the new limits
    plt.savefig(img_path)
    plt.close()


def check_train_fixed_dataset_configs():
    total_indices = 300
    n_per_png = 30
    indices = np.random.randint(0, constants.fixed_dataset_n_configs * constants.fixed_dataset_train_ratio, total_indices)
    data_name = constants.fixed_dataset_name + "_train.pt"
    dataset = torch.load(os.path.join(constants.fixed_dataset_path, data_name))

    saving_path = os.path.join(constants.fixed_dataset_path, "plots")
    os.makedirs(saving_path, exist_ok=True)

    for png_idx in range(total_indices // n_per_png):
        fig, axs = plt.subplots(n_per_png, 5, figsize=(20, 2 * n_per_png))

        for config_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + config_idx
            config = dataset[indices[global_idx]]

            for img_idx in range(5):
                ax = axs[config_idx, img_idx]
                ax.imshow(config[img_idx].detach().cpu().numpy().squeeze(), cmap='gray')
                ax.axis('off')

                if config_idx == 0:
                    titles = ["Initial", "Final", "Easy", "Medium", "Hard"]
                    ax.set_title(titles[img_idx])

        plt.tight_layout()
        plt.savefig(os.path.join(saving_path, f"fixed_configs_set_{png_idx}.png"))
        plt.close(fig)


def main():
    device_manager = DeviceManager()
    dataset = DatasetCreator(device_manager=device_manager)

    check_fixed_dataset_distribution(device_manager.default_device, dataset.dataset_train_path)
    check_fixed_dataset_distribution(device_manager.default_device, dataset.dataset_val_path)
    check_fixed_dataset_distribution(device_manager.default_device, dataset.dataset_test_path)

    check_train_fixed_dataset_configs()


if __name__ == "__main__":
    return_code = main()
    exit(return_code)

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch


sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.paths import DATASET_DIR

from src.gol_adv_sys.DeviceManager import DeviceManager
from src.gol_adv_sys.utils import constants as constants


def check_dataset_distribution(device, saving_path):
    n = constants.grid_size ** 2
    bins = torch.zeros(n+1, dtype=torch.int64, device=device)

    ds_path = {
        "train": DATASET_DIR / f"{constants.dataset_name}_train.pt",
        "val":   DATASET_DIR / f"{constants.dataset_name}_val.pt",
        "test":  DATASET_DIR / f"{constants.dataset_name}_test.pt"
    }

    plot_train = Path(saving_path) / "train_dist.png"
    plot_val   = Path(saving_path) / "val_dist.png"
    plot_test  = Path(saving_path) / "test_dist.png"


    for dataset_path, plot_path in zip([ds_path["train"], ds_path["val"], ds_path["test"]], [plot_train, plot_val, plot_test]):
        dataset = torch.load(dataset_path)

        for config in dataset:
            config_flat = config[0].view(-1)
            living_cells = int(torch.sum(config_flat).item())

            if living_cells <= n:
                bins[living_cells] += 1
            else:
                print(f"Warning: Living cells count {living_cells} exceeds expected range.")

        bins_cpu = bins.to('cpu').numpy()
        max_value = bins_cpu.max()

        # Plotting
        plt.bar(range(n+1), bins_cpu)
        plt.ylim(0, max_value + max_value * 0.1)
        plt.draw()
        plt.savefig(plot_path)
        plt.close()


def check_dataset_configs(saving_path, dataset, metadata, total_indices=200, n_per_png=20):
    indices = np.random.randint(0, len(dataset), total_indices)
    for png_idx in range(total_indices // n_per_png):
        fig, axs = plt.subplots(n_per_png, 7, figsize=(24, 4 * n_per_png))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)

        titles = ["Metadata", "Initial", "Final", "Easy", "Medium", "Hard", "Stable"]
        for config_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + config_idx
            config = dataset[indices[global_idx]]
            meta = metadata[indices[global_idx]]

            for img_idx in range(7):
                ax = axs[config_idx, img_idx]
                if img_idx == 0:
                    ax.text(0, 0.5,
                            f"ID: {meta['id']}\nInitial cells: {meta['n_cells_init']}\nFinal cells: {meta['n_cells_final']}\nPeriod: {meta['period']}\nTransient phase: {meta['transient_phase']}",
                            ha='left', va='center', fontsize=24)
                    ax.axis('off')
                    if config_idx == 0:  # Set titles only for the first row
                        ax.set_title(titles[img_idx], loc='center', fontsize=24)
                else:
                    ax.imshow(config[img_idx - 1].detach().cpu().numpy().squeeze(), cmap='gray')
                    ax.axis('off')
                    if config_idx == 0:  # Set titles only for the first row
                        ax.set_title(titles[img_idx], loc='center', fontsize=24)

        plt.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)

        plt.savefig(Path(saving_path) / f"configs_{png_idx+1}.png", bbox_inches='tight')
        plt.close(fig)


def check_transient_phases(saving_path, metadata):

    pairs = []
    for meta in metadata:
        id = meta["id"]
        transient_phase = meta["transient_phase"]
        pairs.append((id, transient_phase))

    pairs.sort(key=lambda x: x[1])

    ids, transient_phases = zip(*pairs)

    plt.figure(figsize=(10, 6))
    plt.bar(ids, transient_phases, color='blue')
    plt.xlabel("ID")
    plt.ylabel("Transient phase length")
    plt.title("Transient phases")
    plt.grid(True)
    plt.savefig(Path(saving_path) / f"transient_phases.png", bbox_inches='tight')
    plt.close()

def main():
    device_manager = DeviceManager()

    base_saving_path = DATASET_DIR / "plots"
    base_saving_path.mkdir(exist_ok=True)

    check_dataset_distribution(device_manager.default_device, base_saving_path)

    names = ["train", "val", "test"]
    for name in names:

        saving_path = base_saving_path / name
        saving_path.mkdir(exist_ok=True)

        ds_path = DATASET_DIR / f"{constants.dataset_name}_{name}.pt"
        metadata_path = DATASET_DIR / f"{constants.dataset_name}_metadata_{name}.pt"

        dataset = torch.load(ds_path)
        metadata = torch.load(metadata_path)

        check_dataset_configs(saving_path, dataset, metadata)
        check_transient_phases(saving_path, metadata)


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


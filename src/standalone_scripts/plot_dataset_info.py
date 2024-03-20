import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        fig = plt.figure(figsize=(24, 4 * n_per_png))
        gs = GridSpec(n_per_png, 6, figure=fig)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
        titles = ["Initial", "Final", "Easy", "Medium", "Hard", "Stable"]

        for config_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + config_idx
            config = dataset[indices[global_idx]]
            meta = metadata[indices[global_idx]]
            metadata_texts = [
                f"ID: {meta['id']}\nInitial cells: {meta['n_cells_init']}",
                f"Final cells: {meta['n_cells_final']}\nTransient phase: {meta['transient_phase']}\nPeriod: {meta['period']}",
                f"Minimum: {meta['easy_minimum']}\nMaximum: {meta['easy_maximum']}\nQ1: {meta['easy_q1']}\nQ2: {meta['easy_q2']}\nQ3: {meta['easy_q3']}",
                f"Minimum: {meta['medium_minimum']}\nMaximum: {meta['medium_maximum']}\nQ1: {meta['medium_q1']}\nQ2: {meta['medium_q2']}\nQ3: {meta['medium_q3']}",
                f"Minimum: {meta['hard_minimum']}\nMaximum: {meta['hard_maximum']}\nQ1: {meta['hard_q1']}\nQ2: {meta['hard_q2']}\nQ3: {meta['hard_q3']}",
                f"Minimum: {meta['stable_minimum']}\nMaximum: {meta['stable_maximum']}\nQ1: {meta['stable_q1']}\nQ2: {meta['stable_q2']}\nQ3: {meta['stable_q3']}"
            ]

            for img_idx in range(6):
                ax = fig.add_subplot(gs[config_idx, img_idx])
                if img_idx == 0:
                    ax.text(0.5, 0.5, metadata_texts[img_idx], ha='center', va='center', fontsize=14, color='black', wrap=True)
                    ax.set_title(titles[img_idx], loc='center', fontsize=20, pad=20)
                    ax.axis('off')
                else:
                    img_data = config[img_idx].detach().cpu().numpy().squeeze()
                    ax.imshow(img_data, **imshow_kwargs)
                    ax.set_title(titles[img_idx], loc='center', fontsize=20, pad=20)
                    ax.axis('off')
                    ax.text(0, -0.1, metadata_texts[img_idx], ha='left', va='top', fontsize=14, color='black', transform=ax.transAxes, wrap=True)

        # Adjust layout for padding and spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.4)

        # Save and close
        plt.savefig(Path(saving_path) / f"configs_{png_idx+1}.png", bbox_inches='tight')
        plt.close(fig)


# def check_dataset_configs(saving_path, dataset, metadata, total_indices=200, n_per_png=20):
    indices = np.random.randint(0, len(dataset), total_indices)
    for png_idx in range(total_indices // n_per_png):
        fig, axs = plt.subplots(n_per_png, 6, figsize=(24, 4 * n_per_png))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
        titles = ["Initial", "Final", "Easy", "Medium", "Hard", "Stable"]
        for config_idx in range(n_per_png):
            global_idx = png_idx * n_per_png + config_idx
            config = dataset[indices[global_idx]]
            meta = metadata[indices[global_idx]]

            metadata_text_initial = (
                f"ID: {meta['id']}\n"
                f"Initial cells: {meta['n_cells_init']}\n"
            )

            metadata_text_final = (
                f"Final cells: {meta['n_cells_final']}\n"
                f"Transient phase: {meta['transient_phase']}\n"
                f"Period: {meta['period']}\n"
            )

            metadata_text_easy = (
                f"Minimum: {meta['easy_minimum']}\n"
                f"Maximum: {meta['easy_maximum']}\n"
                f"Q1: {meta['easy_q1']}\n"
                f"Q2: {meta['easy_q2']}\n"
                f"Q3: {meta['easy_q3']}\n"
            )

            metadata_text_medium = (
                f"Minimum: {meta['medium_minimum']}\n"
                f"Maximum: {meta['medium_maximum']}\n"
                f"Q1: {meta['medium_q1']}\n"
                f"Q2: {meta['medium_q2']}\n"
                f"Q3: {meta['medium_q3']}\n"
            )

            metadata_text_hard = (
                f"Minimum: {meta['hard_minimum']}\n"
                f"Maximum: {meta['hard_maximum']}\n"
                f"Q1: {meta['hard_q1']}\n"
                f"Q2: {meta['hard_q2']}\n"
                f"Q3: {meta['hard_q3']}\n"
            )

            metadata_text_stable = (
                f"Minimum: {meta['stable_minimum']}\n"
                f"Maximum: {meta['stable_maximum']}\n"
                f"Q1: {meta['stable_q1']}\n"
                f"Q2: {meta['stable_q2']}\n"
                f"Q3: {meta['stable_q3']}\n"
            )

            metadata_text = [metadata_text_initial, metadata_text_final,
                             metadata_text_easy, metadata_text_medium,
                             metadata_text_hard, metadata_text_stable]

            for img_idx in range(6):
                ax = axs[config_idx, img_idx]
                if config_idx % 2 == 0:
                    metadata_text[img_idx]
                    ax.set_title(titles[img_idx], loc='center', fontsize=24)
                    ax.axis('off')
                    ax.text(0.5, 0.5, metadata_text[img_idx], ha='left', fontsize=14, color='black')
                else:
                    ax.imshow(config[img_idx].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                    ax.axis('off')


        plt.tight_layout()
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

    # check_dataset_distribution(device_manager.default_device, base_saving_path)

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


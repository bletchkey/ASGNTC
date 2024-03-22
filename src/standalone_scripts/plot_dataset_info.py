import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.paths import DATASET_DIR
from src.gol_adv_sys.utils import constants as constants


def check_dataset_configs(saving_path, dataset, metadata, total_indices=50, n_per_png=5):

    indices = np.random.randint(0, len(dataset), total_indices)

    for png_idx in range(total_indices // n_per_png):

        fig = plt.figure(figsize=(24, 8 * n_per_png))
        gs = GridSpec(2 * n_per_png, 6, figure=fig, hspace=0, wspace=0.1)

        imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
        titles = ["Initial", "Final", "Easy", "Medium", "Hard", "Stable"]

        for idx_within_page in range(n_per_png):

            global_idx = png_idx * n_per_png + idx_within_page
            config = dataset[indices[global_idx]]
            meta = metadata[indices[global_idx]]

            metadata_texts = [
                f"ID: {meta['id']}\nInitial cells: {meta['n_cells_init']}",
                f"Final cells: {meta['n_cells_final']}\nTransient phase: {meta['transient_phase']}\nPeriod: {meta['period']}",
                f"Minimum: {meta['easy_minimum']:.4f}\nMaximum: {meta['easy_maximum']:.4f}\nQ1: {meta['easy_q1']:.4f}\nQ2: {meta['easy_q2']:.4f}\nQ3: {meta    ['easy_q3']:.4f}",
                f"Minimum: {meta['medium_minimum']:.4f}\nMaximum: {meta['medium_maximum']:.4f}\nQ1: {meta['medium_q1']:.4f}\nQ2: {meta['medium_q2']:.4f}       \nQ3: {meta['medium_q3']:.4f}",
                f"Minimum: {meta['hard_minimum']:.4f}\nMaximum: {meta['hard_maximum']:.4f}\nQ1: {meta['hard_q1']:.4f}\nQ2: {meta['hard_q2']:.4f}\nQ3: {meta    ['hard_q3']:.4f}",
                f"Minimum: {meta['stable_minimum']:.4f}\nMaximum: {meta['stable_maximum']:.4f}\nQ1: {meta['stable_q1']:.4f}\nQ2: {meta['stable_q2']:.4f}       \nQ3: {meta['stable_q3']:.4f}"
            ]

            for col in range(6):
                image_ax = fig.add_subplot(gs[2 * idx_within_page, col])  # Image row
                text_ax = fig.add_subplot(gs[2 * idx_within_page + 1, col])  # Text row

                image_ax.imshow(config[col].detach().cpu().numpy().squeeze(), **imshow_kwargs)
                image_ax.axis('off')
                image_ax.set_title(titles[col], loc='center', fontsize=24,
                                   fontweight='bold', color='black', pad=4)

                text_ax.text(0.05, 0.8, metadata_texts[col], ha='left', va='center', fontsize=14,
                        color='black', wrap=True)
                text_ax.axis('off')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(Path(saving_path) / f"configs_{png_idx+1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def check_transient_phases(saving_path, metadata):
    pairs = [(meta["id"], meta["transient_phase"]) for meta in metadata]
    pairs.sort(key=lambda x: x[1])
    ids, transient_phases = zip(*pairs)
    transient_phases = np.array(transient_phases)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart for transient phases
    axs[0].bar(ids, transient_phases, color='skyblue', edgecolor='black')
    axs[0].set_xlabel("ID", fontsize=12)
    axs[0].set_ylabel("Transient Phase Length", fontsize=12)
    axs[0].set_title("Transient Phases", fontsize=14, fontweight='bold')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(axis='y', which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    axs[0].tick_params(axis='x', rotation=90)

    # Q-Q plot for transient phases
    norm_quantiles = np.linspace(0, 1, len(transient_phases))
    data_quantiles = np.quantile(transient_phases, norm_quantiles)
    theoretical_quantiles = np.quantile(np.random.normal(size=1000), norm_quantiles)

    axs[1].plot(theoretical_quantiles, data_quantiles, 'o', color='skyblue')
    axs[1].plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
                [theoretical_quantiles.min(), theoretical_quantiles.max()],
                'k--')
    axs[1].set_title("Q-Q Plot", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Theoretical Quantiles", fontsize=12)
    axs[1].set_ylabel("Data Quantiles", fontsize=12)

    plt.tight_layout()
    plt.savefig(Path(saving_path) / "transient_phases.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def check_dataset_distribution(saving_path, dataset):
    n = constants.grid_size ** 2
    bins = torch.zeros(n + 1, dtype=torch.int64)

    plot_path = Path(saving_path) / "distribution.png"

    for config in dataset:
        living_cells = torch.sum(config[0].view(-1)).item()
        if living_cells <= n:
            bins[int(living_cells)] += 1
        else:
            print(f"Warning: Living cells count {living_cells} exceeds grid size squared.")

    bins_cpu = bins.numpy()
    max_value = bins_cpu.max()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(n + 1), bins_cpu, color='skyblue', edgecolor='gray')
    plt.title('Distribution of Living Cells')
    plt.xlabel('Number of Living Cells')
    plt.ylabel('Frequency')
    plt.ylim(0, max_value + max_value * 0.1)
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig(plot_path, dpi=300)
    plt.close()


def main():
    base_saving_path = DATASET_DIR / "plots"
    base_saving_path.mkdir(exist_ok=True)

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
        check_dataset_distribution(saving_path, dataset)


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


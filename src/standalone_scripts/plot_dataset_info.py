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
from config.constants import *


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
                (
                    f"ID: {meta[META_ID]}\n"
                    f"Initial cells: {meta[META_N_CELLS_INIT]}"
                ),
                (
                    f"Final cells: {meta[META_N_CELLS_FINAL]}\n"
                    f"Transient phase: {meta[META_TRANSIENT_PHASE]}\n"
                    f"Period: {meta[META_PERIOD]}"
                ),
                (
                    f"Minimum: {meta[META_EASY_MIN]:.4f}\n"
                    f"Maximum: {meta[META_EASY_MAX]:.4f}\n"
                    f"Q1: {meta[META_EASY_Q1]:.4f}\n"
                    f"Q2: {meta[META_EASY_Q2]:.4f}\n"
                    f"Q3: {meta[META_EASY_Q3]:.4f}"
                ),
                (
                    f"Minimum: {meta[META_MEDIUM_MIN]:.4f}\n"
                    f"Maximum: {meta[META_MEDIUM_MAX]:.4f}\n"
                    f"Q1: {meta[META_MEDIUM_Q1]:.4f}\n"
                    f"Q2: {meta[META_MEDIUM_Q2]:.4f}\n"
                    f"Q3: {meta[META_MEDIUM_Q3]:.4f}"
                ),
                (
                    f"Minimum: {meta[META_HARD_MIN]:.4f}\n"
                    f"Maximum: {meta[META_HARD_MAX]:.4f}\n"
                    f"Q1: {meta[META_HARD_Q1]:.4f}\n"
                    f"Q2: {meta[META_HARD_Q2]:.4f}\n"
                    f"Q3: {meta[META_HARD_Q3]:.4f}"
                ),
                (
                    f"Minimum: {meta[META_STABLE_MIN]:.4f}\n"
                    f"Maximum: {meta[META_STABLE_MAX]:.4f}\n"
                    f"Q1: {meta[META_STABLE_Q1]:.4f}\n"
                    f"Q2: {meta[META_STABLE_Q2]:.4f}\n"
                    f"Q3: {meta[META_STABLE_Q3]:.4f}"
                )
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
    pairs = [(meta[META_ID], meta[META_TRANSIENT_PHASE]) for meta in metadata]
    ids, transient_phases = zip(*pairs)
    transient_phases = np.array(transient_phases)

    fig, axs = plt.subplots(2, 2, figsize=(15, 6))

    # Bar chart for transient phases
    axs[0,0].bar(ids, transient_phases, color='#7573d1')
    axs[0,0].set_xlabel("ID", fontsize=12)
    axs[0,0].set_ylabel("Transient Phase Length", fontsize=12)
    axs[0,0].set_title("Transient Phases", fontsize=14, fontweight='bold')
    axs[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0,0].grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[0,0].tick_params(axis='x', rotation=90)

    # ECDF for transient phases
    sorted_phases = np.sort(transient_phases)
    ecdf = np.arange(1, len(sorted_phases) + 1) / len(sorted_phases)
    axs[0, 1].plot(sorted_phases, ecdf, color='#7573d1')
    axs[0, 1].set_xlabel("Transient Phase Length", fontsize=12)
    axs[0, 1].set_ylabel("ECDF", fontsize=12)
    axs[0, 1].set_title("ECDF of Transient Phases", fontsize=14, fontweight='bold')

    # Histogram for transient phases
    axs[1,0].hist(transient_phases, bins=20, color='#7573d1', edgecolor='#3330a1')
    axs[1,0].set_title("Histogram", fontsize=14, fontweight='bold')
    axs[1,0].set_xlabel("Transient Phase Length", fontsize=12)
    axs[1,0].set_ylabel("Frequency", fontsize=12)

    # Boxplot for transient phases
    axs[1,1].boxplot(transient_phases, vert=False, patch_artist=True, boxprops=dict(facecolor='#7573d1'))
    axs[1,1].set_title("Boxplot", fontsize=14, fontweight='bold')
    axs[1,1].set_xlabel("Transient Phase Length", fontsize=12)

    plt.tight_layout()
    plt.savefig(Path(saving_path) / "transient_phases.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def check_dataset_distribution(saving_path, dataset):
    n = GRID_SIZE ** 2
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
    plt.bar(range(n + 1), bins_cpu, color='skyblue')
    plt.title('Distribution of Living Cells')
    plt.xlabel('Number of Living Cells')
    plt.ylabel('Frequency')
    plt.ylim(0, max_value + max_value * 0.1)
    plt.grid(True, linestyle='--', linewidth=1)

    plt.savefig(plot_path, dpi=300)
    plt.close()


def main():
    base_saving_path = DATASET_DIR / "plots"
    base_saving_path.mkdir(exist_ok=True)

    names = ["train", "val", "test"]
    for name in names:

        saving_path = base_saving_path / name
        saving_path.mkdir(exist_ok=True)

        ds_path = DATASET_DIR / f"{DATASET_NAME}_{name}.pt"
        metadata_path = DATASET_DIR / f"{DATASET_NAME}_metadata_{name}.pt"

        dataset = torch.load(ds_path)
        metadata = torch.load(metadata_path)

        check_dataset_configs(saving_path, dataset, metadata)
        check_transient_phases(saving_path, metadata)
        check_dataset_distribution(saving_path, dataset)


if __name__ == "__main__":
    return_code = main()
    exit(return_code)


from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import torch

from src.common.utils.helpers import export_figures_to_pdf
from configs.setup            import setup_base_directory, setup_logging
from configs.paths            import CONFIG_DIR

from src.gol_pred_sys.dataset_manager import DatasetCreator, DatasetManager
from src.common.device_manager        import DeviceManager

from configs.paths import DATASET_DIR
from configs.constants import *


def dataset_creator():
    device_manager = DeviceManager()
    dataset        = DatasetCreator(device_manager=device_manager)
    dataset.create_dataset()
    dataset.save_tensors()


def plot_dataset_samples(saving_path, dataset, metadata, total_indices=50, n_per_png=5):

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
                    f"Initial cells: {meta[META_N_CELLS_INITIAL ]}"
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
        pdf_path = Path(saving_path) / f"configs_{png_idx+1}.pdf"
        export_figures_to_pdf(pdf_path, fig)


def plot_dataset_infos(saving_path: Path, dataset_type: str):

    dataset_manager = DatasetManager()

    transient_phase_infos = dataset_manager.get_infos_transient_phase_per_n_initial_cells(dataset_type)
    period_infos          = dataset_manager.get_infos_period_per_n_initial_cells(dataset_type)

    tp_avg = transient_phase_infos["avg"]
    tp_max = transient_phase_infos["max"]
    tp_min = transient_phase_infos["min"]

    p_avg = period_infos["avg"]
    p_max = period_infos["max"]
    p_min = period_infos["min"]

    fig, axs = plt.subplots(figsize=(10, 12))

    fig.suptitle(f"{dataset_type.capitalize()} Dataset", fontsize=16, fontweight='bold')
    axs.axis('off')

    # Layout
    gs = GridSpec(3, 2, figure=fig)

    dot_size    = 10
    alpha_value = 0.5

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_yticklabels([])
    ax1.scatter(tp_max.keys(), 1, tp_max.values(), s=dot_size, alpha=alpha_value, label="Max")
    ax1.scatter(tp_avg.keys(), 0.5, tp_avg.values(), s=dot_size, alpha=alpha_value, label="Average")
    ax1.scatter(tp_min.keys(), 0, tp_min.values(), s=dot_size, alpha=alpha_value, label="Min")
    ax1.set_title("Transient Phases", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Initial Cells")
    ax1.set_ylabel("Transient Phases Length")
    ax1.legend(["Max", "Average", "Min"], frameon=False)

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_yticklabels([])
    ax2.scatter(p_max.keys(), 1, p_max.values(), s=dot_size, alpha=alpha_value, label="Max")
    ax2.scatter(p_avg.keys(), 0.5, p_avg.values(), s=dot_size, alpha=alpha_value, label="Average")
    ax2.scatter(p_min.keys(), 0, p_min.values(), s=dot_size, alpha=alpha_value, label="Min")
    ax2.set_title("Periods", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Initial Cells")
    ax2.set_ylabel("Period Length")
    ax2.legend(["Max", "Average", "Min"], frameon=False)

    transient_phases = dataset_manager.get_transient_phases(dataset_type)
    transient_phases = np.array(transient_phases)
    sorted_tps       = np.sort(transient_phases)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sorted_tps, np.arange(1, len(sorted_tps) + 1) / len(sorted_tps))
    ax3.set_xlabel("Transient Phase Length")
    ax3.set_ylabel("Empirical Cumulative Distribution Function")

    periods = dataset_manager.get_periods(dataset_type)
    periods = np.array(periods)
    sorted_ps = np.sort(periods)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(sorted_ps, np.arange(1, len(sorted_ps) + 1) / len(sorted_ps))
    ax4.set_xlabel("Period Length")
    ax4.set_ylabel("Empirical Cumulative Distribution Function")

    frequency = dataset_manager.get_frequency_of_n_cells_initial(dataset_type)

    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Distribution of Living Cells")
    ax5.set_xlabel("Number of Living Cells")
    ax5.set_ylabel("Frequency")
    ax5.bar(frequency.keys(), frequency.values(), color='skyblue')

    pdf_path = Path(saving_path) / f"{dataset_type}_infos.pdf"
    export_figures_to_pdf(pdf_path, fig)


def plot_samples():

    base_saving_path = DATASET_DIR / "plots"
    base_saving_path.mkdir(exist_ok=True)

    dataset_type = [TRAIN, VALIDATION, TEST]
    for type in dataset_type:

        saving_path = base_saving_path / type
        saving_path.mkdir(exist_ok=True)

        ds_path       = DATASET_DIR / f"{DATASET_NAME}_{type}.pt"
        metadata_path = DATASET_DIR / f"{DATASET_NAME}_metadata_{type}.pt"

        dataset  = torch.load(ds_path)
        metadata = torch.load(metadata_path)

        plot_dataset_samples(saving_path, dataset, metadata)

        print(f"Dataset {type}: samples plotted.")


def plot_infos():

    base_saving_path = DATASET_DIR / "plots"
    base_saving_path.mkdir(exist_ok=True)

    dataset_type = [TRAIN, VALIDATION, TEST]
    for type in dataset_type:

        saving_path = base_saving_path / type
        saving_path.mkdir(exist_ok=True)

        plot_dataset_infos(saving_path, type)

        print(f"Dataset {type}: infos plotted.")


def main():
    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_dataset.json")

    plot_infos()


if __name__ == "__main__":
    return_code = main()
    exit(return_code)



# def plot_transient_phases(transient_phases, avg_tp_in_cells, saving_path):
#     """
#     Create and save plots of transient phases statistics.
#     """
#     fig, axs = plt.subplots(2, 2, figsize=(12, 12))

#     # Bar chart for average transient phases
#     axs[0, 0].bar(avg_tp_in_cells.keys(), avg_tp_in_cells.values(), color='#7573d1', edgecolor='#3330a1', linewidth=1.5)
#     axs[0, 0].set_title("Average Transient Phase Length by Initial Cells", fontsize=14, fontweight='bold')
#     axs[0, 0].set_xlabel("Number of Initial Cells")
#     axs[0, 0].set_ylabel("Average Transient Phase Length")
#     axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

#     # ECDF plot
#     sorted_phases = np.sort(transient_phases)
#     ecdf = np.arange(1, len(sorted_phases) + 1) / len(sorted_phases)
#     axs[0, 1].plot(sorted_phases, ecdf, color='#7573d1')
#     axs[0, 1].set_title("ECDF of Transient Phases", fontsize=14, fontweight='bold')
#     axs[0, 1].set_xlabel("Transient Phase Length")
#     axs[0, 1].set_ylabel("ECDF")

#     # Histogram of transient phases
#     axs[1, 0].hist(transient_phases, bins=20, color='#7573d1', edgecolor='#3330a1')
#     axs[1, 0].set_title("Histogram of Transient Phases", fontsize=14, fontweight='bold')
#     axs[1, 0].set_xlabel("Transient Phase Length")
#     axs[1, 0].set_ylabel("Frequency")

#     # Boxplot of transient phases
#     axs[1, 1].boxplot(transient_phases, vert=False, patch_artist=True, boxprops=dict(facecolor='#7573d1'))
#     axs[1, 1].set_title("Boxplot of Transient Phases", fontsize=14, fontweight='bold')
#     axs[1, 1].set_xlabel("Transient Phase Length")

#     plt.tight_layout()
#     pdf_path = Path(saving_path) / "transient_phases.pdf"
#     plt.savefig(pdf_path)
#     plt.close()

#     infos = [(meta[META_ID],
#               meta[META_N_CELLS_INITIAL],
#               meta[META_N_CELLS_FINAL],
#               meta[META_PERIOD],
#               meta[META_TRANSIENT_PHASE]) for meta in metadata]

#     ids, n_cells_initial, n_cells_final, periods, transient_phases = zip(*infos)
#     transient_phases = np.array(transient_phases)


#     avg_tp_in_cells = [(0, 0) for _ in range(GRID_SIZE ** 2 + 1)]

#     for i in infos:
#         avg_tp_in_cells[i[1]] = (avg_tp_in_cells[i[1]][0] + i[4], avg_tp_in_cells[i[1]][1] + 1)

#     avg_tp_in_cells = [(i[0] / i[1]) if i[1] != 0 else 0 for i in avg_tp_in_cells]


#     # Plotting
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))

#     # Bar chart for transient phases
#     axs[0,0].bar(range(len(avg_tp_in_cells)), avg_tp_in_cells, color='#7573d1', edgecolor='#3330a1', linewidth=1.5)
#     axs[0,0].set_xlabel("Number of initial Cells", fontsize=12)
#     axs[0,0].set_ylabel("Average Transient Phase Length", fontsize=12)
#     axs[0,0].set_title("Average Transient Phase Length per Initial Cells", fontsize=14, fontweight='bold')
#     axs[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))

#     # ECDF for transient phases
#     sorted_phases = np.sort(transient_phases)
#     ecdf = np.arange(1, len(sorted_phases) + 1) / len(sorted_phases)
#     axs[0, 1].plot(sorted_phases, ecdf, color='#7573d1')
#     axs[0, 1].set_xlabel("Transient Phase Length", fontsize=12)
#     axs[0, 1].set_ylabel("ECDF", fontsize=12)
#     axs[0, 1].set_title("ECDF of Transient Phases", fontsize=14, fontweight='bold')

#     # Histogram for transient phases
#     axs[1,0].hist(transient_phases, bins=20, color='#7573d1', edgecolor='#3330a1')
#     axs[1,0].set_title("Histogram", fontsize=14, fontweight='bold')
#     axs[1,0].set_xlabel("Transient Phase Length", fontsize=12)
#     axs[1,0].set_ylabel("Frequency", fontsize=12)

#     # Boxplot for transient phases
#     axs[1,1].boxplot(transient_phases, vert=False, patch_artist=True, boxprops=dict(facecolor='#7573d1'))
#     axs[1,1].set_title("Boxplot", fontsize=14, fontweight='bold')
#     axs[1,1].set_xlabel("Transient Phase Length", fontsize=12)

#     plt.tight_layout()
#     pdf_path = Path(saving_path) / "transient_phases.pdf"
#     export_figures_to_pdf(pdf_path, fig)

#     DISTRIBUTION

#     n = GRID_SIZE ** 2
#     bins = torch.zeros(n + 1, dtype=torch.int64)

#     for config in dataset:
#         living_cells = torch.sum(config[0].view(-1)).item()
#         if living_cells <= n:
#             bins[int(living_cells)] += 1
#         else:
#             print(f"Warning: Living cells count {living_cells} exceeds grid size squared.")

#     bins_cpu = bins.numpy()
#     max_value = bins_cpu.max()

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(n + 1), bins_cpu, color='skyblue')
#     plt.title('Distribution of Living Cells')
#     plt.xlabel('Number of Living Cells')
#     plt.ylabel('Frequency')
#     plt.ylim(0, max_value + max_value * 0.1)
#     plt.grid(True, linestyle='--', linewidth=1)

#     pdf_path = Path(saving_path) / "distribution.pdf"
#     export_figures_to_pdf(pdf_path, plt.gcf())


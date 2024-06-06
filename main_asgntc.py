from pathlib import Path
import sys
import logging
import os
import gc
import re
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from configs.setup     import setup_base_directory, setup_logging
from configs.constants import *
from configs.paths     import CONFIG_DIR, OUTPUTS_DIR, TRAINED_MODELS_DIR

from src.gol_adv_sys.training_adv import TrainingAdversarial
from src.common.predictor         import Predictor_Baseline, Predictor_ResNet, Predictor_UNet
from src.common.generator         import Generator_DCGAN, Generator_ResGen
from src.gol_adv_sys.utils.eval   import get_generator_eval_stats
from src.common.utils.helpers     import export_figures_to_pdf


def plot_generator_stats(stats):

    fig, axs = plt.subplots(2, 2, figsize=(10, 12))

    # Mapping stats names to proper titles and labels
    titles = {
        "n_cells_initial": "Number of initial cells",
        "n_cells_final": "Number of final cells",
        "period": "Period",
        "transient_phase": "Transient Phase"
    }

    colors = ['#fca903', '#7333d4', '#bf2c69', '#4b8226']

    # Iterate through the axes and plot the corresponding data
    for ax, ((key, title), color) in zip(axs.flat, zip(titles.items(), colors)):
        ax.plot(stats[key], marker='o', linestyle='-', label=title, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()

    # Tight layout to ensure neat spacing between plots
    plt.tight_layout()

    # Save to PDF
    pdf_path = Path(OUTPUTS_DIR) / "generator_stats.pdf"
    export_figures_to_pdf(pdf_path, fig)


def evaluate_generator():
    path  = TRAINED_MODELS_DIR / "adversarial" / "DCGAN_Medium"
    stats = get_generator_eval_stats(path)

    plot_generator_stats(stats)


def train_adversarial():
    # train_adv = TrainingAdversarial(model_p=Predictor_Baseline(TOPOLOGY_TOROIDAL),
    #                                 model_g=Generator_ResGen(32))

    train_adv = TrainingAdversarial(model_p=Predictor_Baseline(TOPOLOGY_TOROIDAL),
                                    model_g=Generator_ResGen(32))

    train_adv.run()


def main():

    setup_base_directory()
    setup_logging(path= CONFIG_DIR / "logging_asgntc.json")

    train_adversarial()
    # evaluate_generator()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)


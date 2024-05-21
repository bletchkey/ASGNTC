import sys
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_pdf(pdf_path, figs):
    """
    Saves one or multiple matplotlib figures to a single PDF file.

    Parameters:
        pdf_path (str): The path to the PDF file where figures will be saved.
        figs (matplotlib.figure.Figure or list of matplotlib.figure.Figure):
            A single matplotlib figure or a list of figures to be saved.

    """
    # Ensure figs is a list
    if not isinstance(figs, list):
        figs = [figs]

    try:
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
        logging.debug(f"Saved {len(figs)} figure(s) to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to save figure(s) to {pdf_path}: {str(e)}")


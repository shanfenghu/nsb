"""
Utilities for creating consistent, publication-quality plots for the NSB paper.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Central Color Palette for consistency across all figures
#    Colors are chosen to be distinct and professional.
NSB_COLORS = {
    'nsb': '#3CB371',                # Main model color (Green)
    'truth': '#2F4F4F',              # Ground truth (Dark Slate Gray)
    'poisson': '#FFA500',            # Poisson baseline (Orange)
    'negative_binomial': '#9370DB',  # Negative Binomial baseline (Medium Purple)
    'softmax_nn': '#A0522D',         # Softmax NN baseline (Sienna)
    'nsb_subcritical': '#4682B4',    # Sub-critical NSB (Steel Blue)
    'fit_line': '#DC143C'            # Fitting line color (Crimson Red)
}

def setup_plot_style():
    """
    Sets the global matplotlib rcParams for a consistent, professional look.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

def save_figure(fig, filename: str, output_dir: str = "figures"):
    """
    Saves a matplotlib figure in multiple high-quality formats.

    Args:
        fig: The matplotlib figure object to save.
        filename (str): The base name for the file (e.g., "synthetic_fits").
        output_dir (str, optional): The directory to save the figures in.
                                      Defaults to "figures".
    """
    # Create the output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define file paths
    pdf_path = output_dir_path / f"{filename}.pdf"
    png_path = output_dir_path / f"{filename}.png"

    # Save in PDF for the paper
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Save in PNG for easy viewing
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Figure saved to '{pdf_path}' and '{png_path}'")


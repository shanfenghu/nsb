"""
Plotting utilities for NSB experiment figures.

This module provides utilities for creating consistent, publication-quality plots.
It includes:
- Centralized color palette (NSB_COLORS) for consistent styling
- Global plot style configuration (setup_plot_style)
- Figure saving utilities (save_figure) for multiple formats

All plotting functions in the experiments/ directory should use these utilities
to ensure visual consistency and professional appearance.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Central Color Palette for consistency across all figures
#    Colors are chosen to be distinct and professional.
NSB_COLORS = {
    'nsb': '#3CB371',                # Main model color (Green)
    'nsb_gru': '#4169E1',            # NSB-GRU (Royal Blue)
    'nsb_lstm': '#00CED1',           # NSB-LSTM (Dark Turquoise)
    'nsb_attention': '#C71585',      # NSB-Attention (Medium Violet Red)
    'truth': '#2F4F4F',              # Ground truth (Dark Slate Gray)
    'poisson': '#FFA500',            # Poisson baseline (Orange)
    'negative_binomial': '#9370DB',  # Negative Binomial baseline (Medium Purple)
    'softmax_nn': '#A0522D',         # Softmax NN baseline (Sienna)
    'softmax_nn_(fair)': '#FF69B4',  # Softmax NN (Fair) baseline (Hot Pink) 
    'nsb_subcritical': '#4682B4',    # Sub-critical NSB (Steel Blue)
    'fit_line': '#DC143C'            # Fitting line color (Crimson Red)
}

def setup_plot_style():
    """
    Sets the global matplotlib rcParams for a consistent, professional look.

    This function configures matplotlib with:
    - Serif fonts for a classic academic appearance
    - STIX math font for LaTeX-style mathematical notation
    - Bold axis labels and titles for emphasis
    - Seaborn whitegrid style for clean backgrounds
    - Appropriate font sizes for publication (14pt base, 12pt ticks, 16pt titles)

    Should be called at the beginning of each plotting script to ensure consistency.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "font.size": 14,
        "mathtext.default": "regular",  # Allow LaTeX-style formatting without full LaTeX
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "lines.linewidth": 2.5,
    })

def save_figure(fig, filename: str, output_dir: str = "figures"):
    """
    Saves a matplotlib figure in multiple high-quality formats.

    This function saves figures in both PDF (vector) and PNG (raster) formats at
    300 DPI with tight bounding boxes. The output
    directory is created if it doesn't exist.

    Args:
        fig: The matplotlib figure object to save
        filename (str): The base name for the file (without extension).
                        Example: "synthetic_fits" → "synthetic_fits.pdf" and "synthetic_fits.png"
        output_dir (str, optional): The directory to save the figures in.
                                      Defaults to "figures". Will be created if missing.
    """
    # Create the output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define file paths
    pdf_path = output_dir_path / f"{filename}.pdf"
    png_path = output_dir_path / f"{filename}.png"

    # Save in PDF (vector format)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Save in PNG for easy viewing
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Figure saved to '{pdf_path}' and '{png_path}'")


"""
This script demonstrates the NSB's unique ability to learn a "gapped"
distribution with probability mass concentrated at both small and very large
integer values, a task impossible for standard models.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson

# --- Local Imports ---
from nsb.model import NSB
from plot_utils import setup_plot_style, NSB_COLORS

def main():
    """
    Train an NSB model on a gapped, bimodal distribution and plot the result.
    """
    print("--- Generating Figure: Gapped Distribution Learning ---")
    
    # --- Configuration ---
    seed = 42
    n_train = 5000
    k_max_plot = 1000 # Plot up to a very large integer
    
    # Define the gapped, bimodal distribution
    dist_params = {
        'mu1': 3,     # First mode (small counts)
        'mu2': 700,   # Second mode (very large counts)
        'w': 0.5     # 99% of data is from the first mode
    }
    
    # --- Generate Data ---
    print("Generating synthetic gapped data...")
    rng = np.random.default_rng(seed)
    comp1 = rng.poisson(dist_params['mu1'], n_train)
    comp2 = rng.poisson(dist_params['mu2'], n_train)
    mask = rng.binomial(1, dist_params['w'], n_train).astype(bool)
    train_data = np.where(mask, comp1, comp2)

    # --- Train NSB Model ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Training NSB model...")
    nsb_model = NSB(hidden_dim=64)
    nsb_model.fit(train_data, epochs=100, batch_size=128)
    
    # --- Generate PMFs for Plotting ---
    k_vals = np.arange(k_max_plot + 1)
    
    # Ground Truth PMF
    true_pmf = dist_params['w'] * poisson.pmf(k_vals, mu=dist_params['mu1']) + \
               (1 - dist_params['w']) * poisson.pmf(k_vals, mu=dist_params['mu2'])
               
    # NSB PMF
    nsb_pmf = nsb_model.predict_pmf(k_max=k_max_plot)
    
    # --- Create the Figure ---
    print("Creating the plot...")
    setup_plot_style()
    # Use a smaller figure size suitable for an inset
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(k_vals, true_pmf, label='Ground Truth', color=NSB_COLORS['truth'], linewidth=2)
    ax.plot(k_vals, nsb_pmf, label='NSB (Ours)', color=NSB_COLORS['nsb'], linestyle='--', linewidth=2)
    
    ax.set_xlabel("Offspring Count (k)", fontsize=10)
    ax.set_ylabel("Probability (log scale)", fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1) # Set y-limit to focus on the relevant probabilities
    ax.legend(fontsize=8)
    
    # --- Save the Figure ---
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "architecture_inset.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    
    print(f"\nInset figure saved to '{output_path}'")

if __name__ == "__main__":
    main()

"""
This script provides direct, quantitative, numerical evidence that the NSB model,
when trained on the real-world `outbreaktrees` dataset, learns a distribution
with a finite mean but an infinite variance, the signature of superspreading.

The script generates:
1.  A figure showing the convergence of the partial sum for the mean and the 
    divergence of the partial sum for the variance, averaged over multiple runs.
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
CONFIG = {
    'seeds': list(range(5)),
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'output_dir_figures': Path("figures"),
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    },
    'unroll_steps': 100000 # Number of steps to unroll for moment calculation
}

# --- Helper Functions ---
def get_pi_sequence(model: NSB, n_steps: int) -> np.ndarray:
    """Unrolls a trained NSB model to get its sequence of break proportions."""
    model.cell.eval()
    with torch.no_grad():
        h = model.h0.cpu()
        pi_sequence = np.zeros(n_steps)
        for k in range(n_steps):
            h, pi_logit = model.cell(h)
            pi_sequence[k] = torch.sigmoid(pi_logit).item()
    return pi_sequence

def calculate_moment_series(pi_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerically computes the partial sums for the mean and variance series."""
    n_steps = len(pi_sequence)
    # P(Y >= k) = product_{i=0}^{k-1} (1 - pi_i)
    log_survival_probs = np.cumsum(np.log(1 - pi_sequence))
    # Prepend 0 for log(P(Y >= 0)) = log(1) = 0
    log_survival_probs = np.insert(log_survival_probs, 0, 0)
    survival_probs = np.exp(log_survival_probs)

    # Mean series: sum_{k=1 to N} P(Y >= k)
    mean_series = np.cumsum(survival_probs[1:])
    
    # Variance series term: k * P(Y >= k)
    variance_terms = np.arange(1, n_steps + 1) * survival_probs[1:]
    variance_series = np.cumsum(variance_terms)
    
    return mean_series, variance_series

# --- Main Experiment Loop ---
def run_experiment():
    """Trains models and computes moment series for each seed."""
    print("--- Running Experiment: Quantitative Moment Analysis ---")
    
    all_mean_series = []
    all_variance_series = []

    # Load the full dataset for training
    df = pd.read_csv(CONFIG['data_path'])
    train_data = df['offspring_count'].values

    for seed in tqdm(CONFIG['seeds'], desc="Running seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
        # Move model to CPU for consistency
        model.cell.to('cpu')
        model.h0 = model.h0.to('cpu')

        model.fit(
            train_data,
            epochs=CONFIG['nn_params']['epochs'],
            lr=CONFIG['nn_params']['lr'],
            batch_size=CONFIG['nn_params']['batch_size']
        )
        
        pi_sequence = get_pi_sequence(model, n_steps=CONFIG['unroll_steps'])
        mean_series, variance_series = calculate_moment_series(pi_sequence)
        
        all_mean_series.append(mean_series)
        all_variance_series.append(variance_series)

    return np.array(all_mean_series), np.array(all_variance_series)

# --- Figure Generation ---
def create_moment_figure(all_mean_series: np.ndarray, all_variance_series: np.ndarray):
    """Generates and saves the moment analysis figure."""
    print("\n--- Generating Figure: Quantitative Moment Analysis ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    n_steps = CONFIG['unroll_steps']
    x_axis = np.arange(1, n_steps + 1)

    # --- Panel (a): Mean Convergence ---
    ax1 = axes[0]
    mean_of_means = np.mean(all_mean_series, axis=0)
    std_of_means = np.std(all_mean_series, axis=0)
    
    ax1.plot(x_axis, mean_of_means, color=NSB_COLORS['nsb'])
    ax1.fill_between(
        x_axis,
        mean_of_means - std_of_means,
        mean_of_means + std_of_means,
        color=NSB_COLORS['nsb'], alpha=0.2
    )
    ax1.set_title("(a) Learned Mean Series", weight='bold')
    ax1.set_xlabel("Number of Terms (N)")
    ax1.set_ylabel("Partial Sum for Mean ($S_\\mu(N)$)")
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--")

    # --- Panel (b): Variance Divergence ---
    ax2 = axes[1]
    mean_of_vars = np.mean(all_variance_series, axis=0)
    std_of_vars = np.std(all_variance_series, axis=0)

    ax2.plot(x_axis, mean_of_vars, color=NSB_COLORS['nsb_subcritical'])
    ax2.fill_between(
        x_axis,
        mean_of_vars - std_of_vars,
        mean_of_vars + std_of_vars,
        color=NSB_COLORS['nsb_subcritical'], alpha=0.2
    )
    ax2.set_title("(b) Learned Variance Series", weight='bold')
    ax2.set_xlabel("Number of Terms (N)")
    ax2.set_ylabel("Partial Sum for Variance ($S_{\\sigma^2}(N)$)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--")
    
    save_figure(fig, "moment_analysis")
    print(mean_of_means[-1], mean_of_vars[-1])

if __name__ == "__main__":
    mean_series_data, variance_series_data = run_experiment()
    create_moment_figure(mean_series_data, variance_series_data)

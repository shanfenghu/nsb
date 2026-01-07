"""
This script provides a deep, quantitative validation of the claim that the
spectral radius of the recurrent weight matrix, rho(W_h), is the mechanism
controlling the model's ability to learn heavy-tailed distributions.

It trains a series of NSB models with their spectral radius capped at
different values and evaluates their performance on both a light-tailed
(Poisson) and a heavy-tailed (Negative Binomial) distribution.

The script generates:
1.  A CSV file with the raw results (`results/spectral_radius_results.csv`).
2.  A figure showing how Test Log-Likelihood and Tail KL Divergence vary as 
    a function of the allowed spectral radius.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, nbinom
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from nsb.constrained_model import ConstrainedNSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
CONFIG = {
    'seeds': list(range(5)),
    'n_train': 5000,
    'n_test': 1000,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'spectral_radii': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    'distributions': {
        'Poisson (light-tailed)': {'type': 'poisson', 'params': {'mu': 3}},
        'Negative Binomial (heavy-tailed)': {'type': 'neg_binomial', 'params': {'n': 2, 'p': 0.1}}
    },
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    },
    'k_max_eval': 400
}

# --- Helper Functions ---
def generate_data(dist_info: dict, n_samples: int, seed: int) -> np.ndarray:
    """Generates a dataset from a specified distribution."""
    rng = np.random.default_rng(seed)
    if dist_info['type'] == 'poisson':
        return rng.poisson(dist_info['params']['mu'], n_samples)
    elif dist_info['type'] == 'neg_binomial':
        return rng.negative_binomial(dist_info['params']['n'], dist_info['params']['p'], n_samples)
    else:
        raise ValueError(f"Unknown distribution type: {dist_info['type']}")

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the KL divergence D_KL(P || Q) for discrete distributions."""
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))

# --- Main Experiment Loop ---
def run_experiment():
    """Runs the spectral radius analysis experiment."""
    print("--- Running Experiment: Spectral Radius Analysis ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for dist_name, dist_info in CONFIG['distributions'].items():
        print(f"\nProcessing Distribution: {dist_name}")
        
        # Generate one fixed dataset for this distribution
        train_data = generate_data(dist_info, CONFIG['n_train'], seed=42)
        test_data = generate_data(dist_info, CONFIG['n_test'], seed=101)

        # Define the "tail" of the distribution
        median_val = int(np.median(train_data))
        tail_start_idx = median_val + 1

        # Pre-calculate the true PMF for KL divergence comparison
        k_vals_eval = np.arange(CONFIG['k_max_eval'] + 1)
        if dist_info['type'] == 'poisson':
            true_pmf = poisson.pmf(k_vals_eval, **dist_info['params'])
        else:
            true_pmf = nbinom.pmf(k_vals_eval, **dist_info['params'])
        
        true_tail_pmf = true_pmf[tail_start_idx:]
        true_tail_pmf /= np.sum(true_tail_pmf)

        for radius in tqdm(CONFIG['spectral_radii'], desc=f"  Testing radii for {dist_name}"):
            for seed in CONFIG['seeds']:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = ConstrainedNSB(
                    hidden_dim=CONFIG['nn_params']['hidden_dim'],
                    max_radius=radius
                )
                model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
                
                log_likelihood = model.log_likelihood(test_data)
                
                learned_pmf = model.predict_pmf(k_max=CONFIG['k_max_eval'])
                learned_tail_pmf = learned_pmf[tail_start_idx:]
                if learned_tail_pmf.sum() > 1e-9:
                    learned_tail_pmf /= np.sum(learned_tail_pmf)
                    tail_kl = kl_divergence(true_tail_pmf, learned_tail_pmf)
                else:
                    tail_kl = np.inf
                
                results.append({
                    'Distribution': dist_name,
                    'Max Radius': radius,
                    'Seed': seed,
                    'Test Log-Likelihood': log_likelihood,
                    'Tail KL Divergence': tail_kl
                })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "spectral_radius_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Figure Generation ---
def create_spectral_figure(results_df: pd.DataFrame):
    """Generates and saves the spectral analysis figure."""
    print("\n--- Generating Figure: Spectral Radius Analysis ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    summary = results_df.groupby(['Distribution', 'Max Radius']).agg(
        ll_mean=('Test Log-Likelihood', 'mean'),
        ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'),
        kl_std=('Tail KL Divergence', 'std')
    ).reset_index()

    colors = {'Poisson (light-tailed)': NSB_COLORS['nsb_subcritical'], 'Negative Binomial (heavy-tailed)': NSB_COLORS['nsb']}
    
    # --- Panel (a): Overall Performance ---
    ax1 = axes[0]
    for dist_name, group in summary.groupby('Distribution'):
        ax1.plot(group['Max Radius'], group['ll_mean'], 'o-', color=colors[dist_name], label=dist_name)
        ax1.fill_between(
            group['Max Radius'],
            group['ll_mean'] - group['ll_std'],
            group['ll_mean'] + group['ll_std'],
            color=colors[dist_name], alpha=0.2
        )
    ax1.set_title("(a) Overall Performance", weight='bold')
    ax1.set_xlabel("Max Spectral Radius ($\u03C1_{max}$)")
    ax1.set_ylabel("Test Log-Likelihood (Higher is better)")
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Criticality Boundary')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    # --- Panel (b): Tail Learning Performance ---
    ax2 = axes[1]
    for dist_name, group in summary.groupby('Distribution'):
        ax2.plot(group['Max Radius'], group['kl_mean'], 'o-', color=colors[dist_name], label=dist_name)
        ax2.fill_between(
            group['Max Radius'],
            group['kl_mean'] - group['kl_std'],
            group['kl_mean'] + group['kl_std'],
            color=colors[dist_name], alpha=0.2
        )
    ax2.set_title("(b) Tail Learning Performance", weight='bold')
    ax2.set_xlabel("Max Spectral Radius ($\u03C1_{max}$)")
    ax2.set_ylabel("Tail KL Divergence (Lower is better)")
    ax2.set_yscale('log')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Criticality Boundary')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    
    save_figure(fig, "spectral_radius_analysis")

if __name__ == "__main__":
    results_df = run_experiment()
    create_spectral_figure(results_df)


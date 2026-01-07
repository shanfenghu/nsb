"""
This script provides a deep, empirical validation of the sample complexity
of the NSB model. It demonstrates how both the overall performance and,
crucially, the model's ability to learn the tail of a distribution improve
as the training dataset size increases.

The script generates:
1.  A CSV file with the raw results (`results/sample_complexity_results.csv`).
2.  A figure showing the learning curves for both overall log-likelihood 
    and the tail KL divergence.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
CONFIG = {
    'seeds': list(range(5)),
    'n_test': 10000,
    'master_n_train': 50000,
    'train_sizes': [100, 500, 1000, 5000, 10000, 50000],
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'distribution': {
        'name': 'Negative Binomial',
        'params': {'n': 2, 'p': 0.1} # Heavy-tailed
    },
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    },
    'k_max_eval': 400 # A high value for accurate KL divergence calculation
}

# --- Helper Functions ---
def generate_data(params: dict, n_samples: int, seed: int) -> np.ndarray:
    """Generates a dataset from a Negative Binomial distribution."""
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(params['n'], params['p'], n_samples)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the KL divergence D_KL(P || Q) for discrete distributions."""
    # Add a small epsilon to avoid log(0) and division by zero
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))

# --- Main Experiment Loop ---
def run_experiment():
    """Runs the sample complexity experiment."""
    print("--- Running Experiment 3: Validating Sample Complexity ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Generate one master training set and one fixed test set
    params = CONFIG['distribution']['params']
    master_train_data = generate_data(params, CONFIG['master_n_train'], seed=42)
    test_data = generate_data(params, CONFIG['n_test'], seed=101)
    
    # --- Define the "tail" of the distribution ---
    # We'll define the tail as anything greater than the median
    median_val = int(np.median(master_train_data))
    tail_start_idx = median_val + 1
    print(f"Defining distribution tail as k > {median_val}")

    # Pre-calculate the true PMF for KL divergence comparison
    true_pmf = nbinom.pmf(np.arange(CONFIG['k_max_eval'] + 1), **params)
    true_tail_pmf = true_pmf[tail_start_idx:]
    true_tail_pmf /= np.sum(true_tail_pmf) # Normalize to be a valid distribution

    for train_size in tqdm(CONFIG['train_sizes'], desc="Processing train sizes"):
        train_subset = master_train_data[:train_size]
        
        for seed in CONFIG['seeds']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
            model.fit(train_subset, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
            
            # 1. Evaluate overall performance
            log_likelihood = model.log_likelihood(test_data)
            
            # 2. Evaluate tail performance
            learned_pmf = model.predict_pmf(k_max=CONFIG['k_max_eval'])
            learned_tail_pmf = learned_pmf[tail_start_idx:]
            learned_tail_pmf /= np.sum(learned_tail_pmf) # Normalize
            
            tail_kl = kl_divergence(true_tail_pmf, learned_tail_pmf)
            
            results.append({
                'Train Size': train_size,
                'Seed': seed,
                'Test Log-Likelihood': log_likelihood,
                'Tail KL Divergence': tail_kl
            })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "sample_complexity_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Figure Generation ---
def create_learning_curve_figure(results_df: pd.DataFrame):
    """Generates and saves the learning curve figure."""
    print("\n--- Generating Figure: Sample Complexity and Learning Curves ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    summary = results_df.groupby('Train Size').agg(
        ll_mean=('Test Log-Likelihood', 'mean'),
        ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'),
        kl_std=('Tail KL Divergence', 'std')
    ).reset_index()

    train_sizes = summary['Train Size']

    # --- Panel (a): Overall Performance ---
    ax1 = axes[0]
    ax1.plot(train_sizes, summary['ll_mean'], 'o-', color=NSB_COLORS['nsb'])
    ax1.fill_between(
        train_sizes,
        summary['ll_mean'] - summary['ll_std'],
        summary['ll_mean'] + summary['ll_std'],
        color=NSB_COLORS['nsb'], alpha=0.2
    )
    ax1.set_title("(a) Overall Performance", weight='bold')
    ax1.set_xlabel("Training Set Size (M)")
    ax1.set_ylabel("Test Log-Likelihood (Higher is better)")
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--")

    # --- Panel (b): Tail Performance ---
    ax2 = axes[1]
    ax2.plot(train_sizes, summary['kl_mean'], 'o-', color=NSB_COLORS['nsb_subcritical'])
    ax2.fill_between(
        train_sizes,
        summary['kl_mean'] - summary['kl_std'],
        summary['kl_mean'] + summary['kl_std'],
        color=NSB_COLORS['nsb_subcritical'], alpha=0.2
    )
    ax2.set_title("(b) Tail Learning Performance", weight='bold')
    ax2.set_xlabel("Training Set Size (M)")
    ax2.set_ylabel("Tail KL Divergence (Lower is better)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--")
    
    save_figure(fig, "sample_complexity_curves")

if __name__ == "__main__":
    results_df = run_experiment()
    create_learning_curve_figure(results_df)


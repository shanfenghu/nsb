"""
This script analyzes the sensitivity of the NSB model's performance to the
size of its recurrent hidden layer (hidden_dim). It demonstrates that the
model's superiority is robust across a range of complexities and illustrates
the trade-off between model size and performance.

The script generates:
1.  A CSV file with the raw results (`results/hidden_size_sensitivity.csv`).
2.  A line plot showing Test Log-Likelihood vs. the number of model parameters, 
    annotated with the hidden dimension size.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS
from utils import count_parameters

# --- Configuration ---
CONFIG = {
    'seeds': list(range(5)),
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'test_size': 0.2,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'hidden_sizes': [4, 8, 16, 32, 64, 128],
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
    }
}

# --- Helper Function ---
def load_and_split_data(path: Path, test_size: float, seed: int):
    """Loads the processed count data and splits it into train/test sets."""
    df = pd.read_csv(path)
    counts = df['offspring_count'].values
    return train_test_split(counts, test_size=test_size, random_state=seed)

# --- Main Experiment Loop ---
def run_experiment():
    """Runs the hidden size sensitivity experiment."""
    print("--- Running Experiment: NSB Hidden Size Sensitivity ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Use a fixed train/test split for all runs to ensure comparability
    train_data, test_data = load_and_split_data(CONFIG['data_path'], CONFIG['test_size'], seed=42)

    for hidden_dim in tqdm(CONFIG['hidden_sizes'], desc="Processing hidden sizes"):
        for seed in CONFIG['seeds']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = NSB(hidden_dim=hidden_dim)
            
            model.fit(
                train_data,
                epochs=CONFIG['nn_params']['epochs'],
                lr=CONFIG['nn_params']['lr'],
                batch_size=CONFIG['nn_params']['batch_size']
            )
            
            log_likelihood = model.log_likelihood(test_data)
            num_params = count_parameters(model.cell) + model.h0.numel()
            
            results.append({
                'Hidden Size': hidden_dim,
                'Seed': seed,
                'Test Log-Likelihood': log_likelihood,
                'Num. Params': num_params
            })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "hidden_size_sensitivity.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Figure Generation ---
def create_sensitivity_figure(results_df: pd.DataFrame):
    """Generates and saves the sensitivity analysis figure."""
    print("\n--- Generating Figure: Hidden Size Sensitivity ---")
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    summary = results_df.groupby('Hidden Size').agg(
        ll_mean=('Test Log-Likelihood', 'mean'),
        ll_std=('Test Log-Likelihood', 'std'),
        params=('Num. Params', 'first')
    ).reset_index()

    ax.plot(summary['params'], summary['ll_mean'], 'o-', color=NSB_COLORS['nsb'])
    ax.fill_between(
        summary['params'],
        summary['ll_mean'] - summary['ll_std'],
        summary['ll_mean'] + summary['ll_std'],
        color=NSB_COLORS['nsb'], alpha=0.2
    )
    
    # Annotate each point with its hidden dimension
    for i, row in summary.iterrows():
        ax.text(row['params'], row['ll_mean'] + 0.005, f"d$_h$={row['Hidden Size']}",
                horizontalalignment='center', fontsize=10)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Log-Likelihood")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--")
    
    save_figure(fig, "hidden_size_sensitivity")

if __name__ == "__main__":
    results_df = run_experiment()
    create_sensitivity_figure(results_df)
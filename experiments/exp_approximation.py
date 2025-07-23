"""
Experiment: Validating Expressiveness.

This script runs a comprehensive validation of the NSB model's ability to
learn a diverse suite of synthetic distributions. It trains the NSB model and
all baselines on five different ground-truth distributions and evaluates their
performance.

The script generates:
1.  A CSV file with the raw results of all runs (`results/synthetic_approximation_results.csv`).
2.  A LaTeX-formatted performance table for the paper (Table 1).
3.  A multi-panel figure comparing the distributional fits (Figure 2).
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, binom, nbinom
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from nsb.poisson import PoissonMLE
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.softmax_nn import SoftmaxNN
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
# CONFIG = {
#     'seeds': list(range(10)),
#     'n_train': 50000,
#     'n_test': 10000,
#     'output_dir_results': Path("results"),
#     'output_dir_figures': Path("figures"),
#     'distributions': {
#         'Poisson': {'mu': 3},
#         'Binomial': {'n': 20, 'p': 0.2},
#         'Negative Binomial': {'n': 5, 'p': 0.5},
#         'Mixture': {'mu1': 2, 'mu2': 15, 'w': 0.5},
#         'Zero-Inflated': {'mu': 3, 'omega': 0.5} # 50% extra zeros
#     },
#     'nn_params': {
#         'epochs': 100,
#         'lr': 1e-3,
#         'batch_size': 128,
#         'hidden_dim': 64,
#         'k_max': 150 # For SoftmaxNN
#     }
# }

CONFIG = {
    'seeds': list(range(2)),
    'n_train': 500,
    'n_test': 100,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'distributions': {
        'Poisson': {'mu': 3},
        'Binomial': {'n': 20, 'p': 0.2},
        'Negative Binomial': {'n': 2, 'p': 0.1},
        'Mixture': {'mu1': 5, 'mu2': 40, 'w': 0.6},
        'Zero-Inflated': {'mu': 8, 'omega': 0.6}
    },
    'nn_params': {
        'epochs': 2,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
        'k_max': 150 # For SoftmaxNN
    }
}

def zipois_pmf(k, mu, omega):
    """PMF of the Zero-Inflated Poisson distribution."""
    if omega < 0 or omega > 1:
        raise ValueError("omega (zero-inflation) must be in [0, 1]")
    
    # Calculate Poisson PMF
    poisson_pmf_vals = poisson.pmf(k, mu)
    
    # Apply zero-inflation
    pmf = (1 - omega) * poisson_pmf_vals
    
    # Handle the k=0 case specifically
    if isinstance(k, (int, float)) and k == 0:
        pmf = omega + (1 - omega) * poisson_pmf_vals
    elif isinstance(k, np.ndarray):
        pmf[k == 0] = omega + (1 - omega) * poisson_pmf_vals[k == 0]
        
    return pmf

# --- Data Generation ---
def generate_data(dist_name: str, params: dict, n_samples: int, seed: int) -> np.ndarray:
    """Generates a dataset from a specified distribution."""
    rng = np.random.default_rng(seed)
    if dist_name == 'Poisson':
        return rng.poisson(params['mu'], n_samples)
    elif dist_name == 'Binomial':
        return rng.binomial(params['n'], params['p'], n_samples)
    elif dist_name == 'Negative Binomial':
        return rng.negative_binomial(params['n'], params['p'], n_samples)
    elif dist_name == 'Mixture':
        comp1 = rng.poisson(params['mu1'], n_samples)
        comp2 = rng.poisson(params['mu2'], n_samples)
        mask = rng.binomial(1, params['w'], n_samples).astype(bool)
        return np.where(mask, comp1, comp2)
    elif dist_name == 'Zero-Inflated':
        # Custom implementation for zipois
        counts = rng.poisson(params['mu'], n_samples)
        mask = rng.binomial(1, params['omega'], n_samples).astype(bool)
        counts[mask] = 0
        return counts
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

# --- Main Experiment Loop ---
def run_experiments():
    """Runs the full suite of synthetic experiments."""
    print("--- Running Experiment 1: Validating Expressiveness ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for dist_name, params in CONFIG['distributions'].items():
        print(f"\nProcessing Distribution: {dist_name}")
        
        # Generate one fixed dataset for this distribution
        train_data = generate_data(dist_name, params, CONFIG['n_train'], seed=42)
        test_data = generate_data(dist_name, params, CONFIG['n_test'], seed=101)

        for seed in tqdm(CONFIG['seeds'], desc=f"  Running seeds for {dist_name}"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # --- Train and Evaluate Models ---
            models = {
                'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
                'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=CONFIG['nn_params']['hidden_dim']),
                'Negative Binomial': NegativeBinomialMLE(),
                'Poisson': PoissonMLE()
            }

            for model_name, model in models.items():
                try:
                    # Train model
                    if hasattr(model, 'fit'):
                        if isinstance(model, (NSB, SoftmaxNN)):
                            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'], batch_size=CONFIG['nn_params']['batch_size'])
                        else:
                            model.fit(train_data)
                    
                    # Evaluate model
                    log_likelihood = model.log_likelihood(test_data)
                
                except (ValueError, RuntimeError) as e:
                    # Handle cases where a model cannot be fitted (e.g., NB on underdispersed data)
                    print(f"    - WARNING: Could not fit {model_name} on {dist_name}. Reason: {e}")
                    log_likelihood = -np.inf

                results.append({
                    'Distribution': dist_name,
                    'Model': model_name,
                    'Seed': seed,
                    'Test Log-Likelihood': log_likelihood
                })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "synthetic_approximation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Table and Figure Generation ---
def create_results_table(results_df: pd.DataFrame):
    """Generates and prints the LaTeX table for the paper."""
    summary = results_df.groupby(['Distribution', 'Model'])['Test Log-Likelihood'].agg(['mean', 'std']).reset_index()
    
    pivot_table = summary.pivot(index='Distribution', columns='Model', values=['mean', 'std'])
    
    # Reorder for the paper
    dist_order = ['Poisson', 'Binomial', 'Negative Binomial', 'Mixture', 'Zero-Inflated']
    model_order = ['NSB (Ours)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    pivot_table = pivot_table.reindex(dist_order)
    pivot_table = pivot_table.reindex(columns=model_order, level='Model')

    # Find the best model in each row
    mean_table = pivot_table['mean']
    best_indices = mean_table.idxmax(axis=1)
    
    # Format for LaTeX
    latex_str = "\\begin{tabular}{l" + "c" * len(model_order) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "Distribution & " + " & ".join(model_order) + " \\\\\n"
    latex_str += "\\midrule\n"

    for dist in dist_order:
        latex_str += f"{dist}"
        for model in model_order:
            mean = pivot_table.loc[dist, ('mean', model)]
            std = pivot_table.loc[dist, ('std', model)]
            
            cell_str = f"${mean:.2f} \\pm {std:.2f}$"
            if model == best_indices[dist]:
                cell_str = f"\\textbf{{{cell_str}}}"
            latex_str += f" & {cell_str}"
        latex_str += " \\\\\n"
        
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}"

    # Save the LaTeX table to a file
    table_path = CONFIG['output_dir_results'] / "synthetic_approximation_results_table.tex"
    with open(table_path, 'w') as f:
        f.write(latex_str)
    
    print("\n--- LaTeX Table 1: Performance on Synthetic Data ---")
    print(f"LaTeX table saved to '{table_path}'")

def create_comparison_figure():
    """Generates and saves the visual comparison figure for the paper."""
    print("\n--- Generating Figure 2: Visual Validation ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    dists_to_plot = {
        'Negative Binomial': axes[0],
        'Mixture': axes[1],
        'Zero-Inflated': axes[2]
    }

    for dist_name, ax in dists_to_plot.items():
        params = CONFIG['distributions'][dist_name]
        train_data = generate_data(dist_name, params, CONFIG['n_train'], seed=42)
        
        # --- Get Ground Truth PMF ---
        # For the Negative Binomial plot, we need to extend the x-axis
        # beyond the Softmax NN's k_max to show its truncation.
        if dist_name == 'Negative Binomial':
            k_max_plot = CONFIG['nn_params']['k_max'] + 20 # Plot up to 160
        else:
            k_max_plot = int(np.percentile(train_data, 99.9))

        k_vals = np.arange(k_max_plot + 1)
        if dist_name == 'Negative Binomial':
            true_pmf = nbinom.pmf(k_vals, n=params['n'], p=params['p'])
        elif dist_name == 'Mixture':
            true_pmf = params['w'] * poisson.pmf(k_vals, mu=params['mu1']) + \
                       (1 - params['w']) * poisson.pmf(k_vals, mu=params['mu2'])
        elif dist_name == 'Zero-Inflated':
            # true_pmf = zipois(mu=params['mu'], omega=params['omega']).pmf(k_vals)
            true_pmf = zipois_pmf(k_vals, mu=params['mu'], omega=params['omega'])

        ax.plot(k_vals, true_pmf, 'o-', color=NSB_COLORS['truth'], label='Ground Truth', markersize=4, zorder=5)

        # --- Train and Plot Models ---
        models = {
            'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
            'Poisson': PoissonMLE(),
            'Negative Binomial': NegativeBinomialMLE(),
            'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'])
        }
        
        for model_name, model in models.items():
            try:
                if isinstance(model, (NSB, SoftmaxNN)):
                    model.fit(train_data, epochs=50) # Fewer epochs for figure generation
                else:
                    model.fit(train_data)

                if isinstance(model, SoftmaxNN):
                    # SoftmaxNN has a fixed k_max; its pmf may be shorter or longer
                    # than the plotting range, so we adjust its length.
                    pmf_full = model.predict_pmf()
                    pmf = np.zeros(k_max_plot + 1)
                    len_to_copy = min(len(pmf_full), len(pmf))
                    pmf[:len_to_copy] = pmf_full[:len_to_copy]
                elif hasattr(model, 'predict_pmf'): # This will now only be the NSB
                    pmf = model.predict_pmf(k_max=k_max_plot)
                else: # For Poisson and NegativeBinomial
                    pmf = model.pmf(k_vals)
                
                ax.plot(k_vals, pmf, '--', color=NSB_COLORS[model_name.lower().replace(' (ours)','').replace(' ','_')], label=model_name, alpha=0.8)
            except Exception as e:
                print(f"Could not plot {model_name} for {dist_name}: {e}")

        ax.set_title(f"(b) {dist_name} Fit", weight='bold')
        ax.set_xlabel("Offspring Count (k)")
        if dist_name == 'Negative Binomial':
            ax.set_yscale('log')
            ax.set_ylabel("Probability (log scale)")
        else:
            ax.set_ylabel("Probability")
        ax.legend()

    save_figure(fig, "synthetic_fits")


if __name__ == "__main__":
    results_df = run_experiments()
    create_results_table(results_df)
    create_comparison_figure()

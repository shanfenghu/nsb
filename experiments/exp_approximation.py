"""
Experiment: Validating Expressiveness.

This script runs a comprehensive validation of the NSB model's ability to
learn a diverse suite of synthetic distributions. It trains the NSB model and
all baselines on five different ground-truth distributions and evaluates their
performance.

The script generates:
1.  A CSV file with the raw results of all runs (`results/synthetic_approximation_results.csv`).
2.  A LaTeX-formatted performance table for the paper.
3.  A multi-panel figure comparing the distributional fits.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, binom, nbinom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from nsb.poisson import PoissonMLE
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.softmax_nn import SoftmaxNN
from plot_utils import setup_plot_style, save_figure, NSB_COLORS
from utils import count_parameters

# --- Configuration ---
CONFIG = {
    'seeds': list(range(5)),
    'n_train': 5000,
    'n_test': 1000,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'distributions': {
        'Poisson': {'mu': 3},
        'Binomial': {'n': 20, 'p': 0.2},
        'Negative Binomial': {'n': 2, 'p': 0.1},
        'Mixture': {'mu1': 5, 'mu2': 40, 'w': 0.6},
        'Zero-Inflated': {'mu': 8, 'omega': 0.6},
        'Zero-Inflated NB': {'n': 2, 'p': 0.1, 'omega': 0.5}
    },
    'nn_params': {
        'epochs': 50,
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

def zinb_pmf(k, n, p, omega):
    """PMF of the Zero-Inflated Negative Binomial distribution."""
    if omega < 0 or omega > 1:
        raise ValueError("omega (zero-inflation) must be in [0, 1]")
    
    nb_pmf_vals = nbinom.pmf(k, n=n, p=p)
    pmf = (1 - omega) * nb_pmf_vals
    
    if isinstance(k, (int, float)) and k == 0:
        pmf = omega + (1 - omega) * nb_pmf_vals
    elif isinstance(k, np.ndarray):
        pmf[k == 0] = omega + (1 - omega) * nb_pmf_vals[k == 0]
        
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
    elif dist_name == 'Zero-Inflated NB':
        counts = rng.negative_binomial(params['n'], params['p'], n_samples)
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

            # Define the hidden dimension for the fair baseline
            fair_hidden_dim = 27 # Calculated to match NSB's ~4.3k params

            # --- Train and Evaluate Models ---
            models = {
                'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
                'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=CONFIG['nn_params']['hidden_dim']),
                'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim),
                'Negative Binomial': NegativeBinomialMLE(),
                'Poisson': PoissonMLE()
            }

            for model_name, model in models.items():
                # --- Count Parameters ---
                if isinstance(model, NSB):
                    # The NSB's nn.Module is the 'cell' plus the initial hidden state 'h0'
                    num_params = count_parameters(model.cell) + model.h0.numel()
                elif isinstance(model, SoftmaxNN):
                    num_params = count_parameters(model.model)
                elif isinstance(model, NegativeBinomialMLE):
                    num_params = 2
                elif isinstance(model, PoissonMLE):
                    num_params = 1
                else:
                    num_params = 0

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
                    'Test Log-Likelihood': log_likelihood,
                    'Num. Params': num_params
                })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "synthetic_approximation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Table and Figure Generation ---
def create_results_table(results_df: pd.DataFrame):
    """Generates and saves the LaTeX table for the paper."""
    # Get the parameter counts (they are the same for each run)
    param_counts = results_df.groupby('Model')['Num. Params'].first()

    summary = results_df.groupby(['Distribution', 'Model'])['Test Log-Likelihood'].agg(['mean', 'std']).reset_index()
    
    pivot_table = summary.pivot(index='Distribution', columns='Model', values=['mean', 'std'])
    
    dist_order = ['Poisson', 'Binomial', 'Negative Binomial', 'Mixture', 'Zero-Inflated']
    # Add the new fair baseline to the model order
    model_order = ['NSB (Ours)', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    
    pivot_table = pivot_table.reindex(dist_order)
    pivot_table = pivot_table.reindex(columns=model_order, level='Model')

    mean_table = pivot_table['mean']
    best_indices = mean_table.idxmax(axis=1)
    
    # Add a new column for Num. Params in the header
    latex_str = "\\begin{tabular}{l|c|" + "c" * len(model_order) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "Distribution & Num. Params & " + " & ".join(model_order) + " \\\\\n"
    latex_str += "\\midrule\n"

    for dist in dist_order:
        # This part is a bit tricky, we'll just add the params for the first model in the row
        pass

    # Create a new header with parameter counts
    header_with_params = []
    for model in model_order:
        params = param_counts.get(model, 0)
        header_with_params.append(f"{model} ({params:,} p.)")

    latex_str = "\\begin{tabular}{l" + "c" * len(model_order) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "Distribution & " + " & ".join(header_with_params) + " \\\\\n"
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

def create_comparison_figure(results_df: pd.DataFrame):
    """
    Generates and saves the visual comparison figure for the paper.
    The figure is a direct visualization of the results from the first seed.
    """
    print("\n--- Generating Visual Validation ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Use the results from the FIRST seed to generate a representative plot
    plot_seed = CONFIG['seeds'][0]
    plot_results = results_df[results_df['Seed'] == plot_seed]

    dists_to_plot = {
        'Negative Binomial': axes[0],
        'Mixture': axes[1],
        # 'Zero-Inflated': axes[2]
        'Zero-Inflated NB': axes[2]
    }

    for i, (dist_name, ax) in enumerate(dists_to_plot.items()):
        params = CONFIG['distributions'][dist_name]
        # Generate the same fixed training data used in the main experiment
        train_data = generate_data(dist_name, params, CONFIG['n_train'], seed=42)

        # --- Get Ground Truth PMF ---
        if dist_name == 'Negative Binomial':
            k_max_plot = CONFIG['nn_params']['k_max'] + 25 # Plot beyond k_max
        else:
            k_max_plot = int(np.percentile(train_data, 99.9))
        
        k_vals = np.arange(k_max_plot + 1)
        
        if dist_name == 'Negative Binomial':
            true_pmf = nbinom.pmf(k_vals, n=params['n'], p=params['p'])
        elif dist_name == 'Mixture':
            true_pmf = params['w'] * poisson.pmf(k_vals, mu=params['mu1']) + \
                       (1 - params['w']) * poisson.pmf(k_vals, mu=params['mu2'])
        elif dist_name == 'Zero-Inflated NB':
            true_pmf = zinb_pmf(k_vals, n=params['n'], p=params['p'], omega=params['omega'])
        # elif dist_name == 'Zero-Inflated':
        #     true_pmf = zipois_pmf(k_vals, mu=params['mu'], omega=params['omega'])

        ax.plot(k_vals, true_pmf, 'o-', color=NSB_COLORS['truth'], label='Ground Truth', markersize=4, zorder=5)

        # --- Train and Plot Models from the chosen seed ---
        fair_hidden_dim = 27
        models = {
            'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
            'Poisson': PoissonMLE(),
            'Negative Binomial': NegativeBinomialMLE(),
            'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max']),
            'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim)
        }
        
        for model_name, model in models.items():
            try:
                # Get the specific log-likelihood for this seed from the results
                log_likelihood_series = plot_results[
                    (plot_results['Distribution'] == dist_name) & 
                    (plot_results['Model'] == model_name)
                ]['Test Log-Likelihood']
                
                if log_likelihood_series.empty or not np.isfinite(log_likelihood_series.iloc[0]):
                    print(f"Skipping plot for {model_name} on {dist_name} due to invalid result.")
                    continue

                log_likelihood = log_likelihood_series.iloc[0]
                legend_label = f"{model_name} (LL: {log_likelihood:.2f})"

                # Re-train the model with the specific seed to get the exact curve
                torch.manual_seed(plot_seed)
                np.random.seed(plot_seed)
                
                if isinstance(model, (NSB, SoftmaxNN)):
                    model.fit(train_data, epochs=50)
                else:
                    model.fit(train_data)
                
                # Generate the PMF for plotting
                if isinstance(model, SoftmaxNN):
                    pmf_full = model.predict_pmf()
                    pmf = np.zeros(k_max_plot + 1)
                    len_to_copy = min(len(pmf_full), len(pmf))
                    pmf[:len_to_copy] = pmf_full[:len_to_copy]
                elif hasattr(model, 'predict_pmf'):
                    pmf = model.predict_pmf(k_max=k_max_plot)
                else:
                    pmf = model.pmf(k_vals)
                
                color_key = model_name.lower().replace(' (ours)','').replace(' ','_')
                ax.plot(k_vals, pmf, '--', color=NSB_COLORS[color_key], label=legend_label, alpha=0.9)

            except Exception as e:
                print(f"Could not plot {model_name} for {dist_name}: {e}")

        ax.set_title(f"({chr(97 + i)}) {dist_name} Fit", weight='bold')
        ax.set_xlabel("Offspring Count (k)")
        if dist_name == 'Negative Binomial':
            ax.set_yscale('log')
            ax.set_ylabel("Probability (log scale)")
        else:
            ax.set_ylabel("Probability")
        ax.legend()

    ax_nb = axes[0] # The Negative Binomial plot is the first one
    dist_name_nb = 'Negative Binomial'
    params_nb = CONFIG['distributions'][dist_name_nb]
    train_data_nb = generate_data(dist_name_nb, params_nb, CONFIG['n_train'], seed=42)
    plot_seed = CONFIG['seeds'][0]

    # Create an inset on the top right
    axins = ax_nb.inset_axes([0.1, 0.45, 0.38, 0.38])

    # Plot the far tail in the inset
    k_max_inset = 1000
    k_vals_inset = np.arange(k_max_inset + 1)

    # Ground truth for inset
    true_pmf_inset = nbinom.pmf(k_vals_inset, n=params_nb['n'], p=params_nb['p'])
    cutoff = CONFIG['nn_params']['k_max'] + 1
    true_tail_mass = 1.0 - np.sum(true_pmf_inset[:cutoff])
    axins.plot(k_vals_inset, true_pmf_inset, 'o-', color=NSB_COLORS['truth'], label=f'GT (T.M.: {true_tail_mass:.2e})', markersize=2, linewidth=1.5)

    # NSB model for inset
    nsb_model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
    torch.manual_seed(plot_seed)
    np.random.seed(plot_seed)
    nsb_model.fit(train_data_nb, epochs=50)
    nsb_pmf_inset = nsb_model.predict_pmf(k_max=k_max_inset)
    nsb_tail_mass = 1 - np.sum(nsb_pmf_inset[:cutoff])
    axins.plot(k_vals_inset, nsb_pmf_inset, '--', color=NSB_COLORS['nsb'], label=f'NSB (T.M.: {nsb_tail_mass:.2e})', linewidth=1.5)

    # Style the inset
    axins.set_yscale('log')
    axins.set_title("Far Tail (k > 150)", fontsize=10)
    axins.set_xlim(150, k_max_inset) # Focus on the far tail beyond SoftmaxNN's limit
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)
    axins.legend(fontsize=8)

    # --- Add a special inset for the Zero-Inflated NB plot ---
    ax_zinb = axes[2] # The ZINB plot is the third one
    dist_name_zinb = 'Zero-Inflated NB'
    params_zinb = CONFIG['distributions'][dist_name_zinb]
    train_data_zinb = generate_data(dist_name_zinb, params_zinb, CONFIG['n_train'], seed=42)
    plot_seed = CONFIG['seeds'][0]

    # Create an inset
    axins_zinb = ax_zinb.inset_axes([0.5, 0.15, 0.38, 0.38])

    # Plot the far tail in the inset
    k_max_inset = 1000
    k_vals_inset = np.arange(k_max_inset + 1)

    # Ground truth for inset
    true_pmf_inset = zinb_pmf(k_vals_inset, n=params_zinb['n'], p=params_zinb['p'], omega=params_zinb['omega'])
    true_tail_mass_zinb = 1.0 - np.sum(true_pmf_inset[:cutoff])
    axins_zinb.plot(k_vals_inset, true_pmf_inset, 'o-', color=NSB_COLORS['truth'], label=f'GT (T.M.: {true_tail_mass_zinb:.2e})', markersize=2, linewidth=1.5)

    # NSB model for inset
    nsb_model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
    torch.manual_seed(plot_seed)
    np.random.seed(plot_seed)
    nsb_model.fit(train_data_zinb, epochs=50)
    nsb_pmf_inset = nsb_model.predict_pmf(k_max=k_max_inset)
    nsb_tail_mass_zinb = 1 - np.sum(nsb_pmf_inset[:cutoff])
    axins_zinb.plot(k_vals_inset, nsb_pmf_inset, '--', color=NSB_COLORS['nsb'], label=f'NSB (T.M.: {nsb_tail_mass_zinb:.2e})', linewidth=1.5)

    # Style the inset
    axins_zinb.set_yscale('log')
    axins_zinb.set_title("Far Tail (k > 150)", fontsize=10)
    axins_zinb.set_xlim(150, k_max_inset)
    axins_zinb.tick_params(axis='x', labelsize=8)
    axins_zinb.tick_params(axis='y', labelsize=8)
    axins_zinb.legend(fontsize=8)

    save_figure(fig, "synthetic_fits")

if __name__ == "__main__":
    results_df = run_experiments()
    create_results_table(results_df)
    create_comparison_figure(results_df)

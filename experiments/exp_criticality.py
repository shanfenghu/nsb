"""
This script provides direct empirical evidence for the claim that the spectral
radius of the recurrent weight matrix, rho(W_h), is the mechanism that
controls the model's ability to learn heavy-tailed distributions.

The script generates:
1.  A CSV file with the raw results (`results/criticality_results.csv`).
2.  A two-panel figure for the paper (Figure 3) comparing the distributional
    fits and the spectral properties of an unconstrained, a sub-critical,
    and a critical NSB model.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, nbinom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from nsb.constrained_model import ConstrainedNSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
CONFIG = {
    'seeds': list(range(2)),
    'n_train': 500,
    'n_test': 100,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    # 'distribution': {
    #     'name': 'Negative Binomial',
    #     'params': {'n': 2, 'p': 0.1} # Heavy-tailed
    # },
    'distribution': {
        'name': 'Pathological',
        'params': {'mu': 2, 'n': 2, 'p': 0.05, 'w': 0.5, 'omega': 0.3}
    },
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    }
}

# --- Helper Functions ---
def pathological_pmf(k, mu, n, p, w, omega):
    """PMF of the Zero-Inflated Mixture of Poisson and Negative Binomial."""
    poisson_comp = poisson.pmf(k, mu)
    nb_comp = nbinom.pmf(k, n, p)
    mixture_pmf = w * poisson_comp + (1 - w) * nb_comp
    pmf = (1 - omega) * mixture_pmf
    if isinstance(k, (int, float)) and k == 0:
        pmf = omega + (1 - omega) * mixture_pmf
    elif isinstance(k, np.ndarray):
        pmf[k == 0] = omega + (1 - omega) * mixture_pmf[k == 0]
    return pmf

# def generate_data(params: dict, n_samples: int, seed: int) -> np.ndarray:
#     """Generates a dataset from a Negative Binomial distribution."""
#     rng = np.random.default_rng(seed)
#     return rng.negative_binomial(params['n'], params['p'], n_samples)

def generate_data(params: dict, n_samples: int, seed: int) -> np.ndarray:
    """Generates a dataset from the Pathological distribution."""
    rng = np.random.default_rng(seed)
    # Generate the mixture component
    poisson_samples = rng.poisson(params['mu'], n_samples)
    nb_samples = rng.negative_binomial(params['n'], params['p'], n_samples)
    mask = rng.binomial(1, params['w'], n_samples).astype(bool)
    counts = np.where(mask, poisson_samples, nb_samples)
    # Apply zero-inflation
    zero_mask = rng.binomial(1, params['omega'], n_samples).astype(bool)
    counts[zero_mask] = 0
    return counts

def get_eigenvalues(model: NSB) -> np.ndarray:
    """Calculates the eigenvalues of the recurrent weight matrix W_h."""
    weight_matrix = model.cell.fc_h.weight.data.cpu().numpy()
    return np.linalg.eigvals(weight_matrix)

# --- Main Experiment Loop ---
def run_experiment():
    """Runs the criticality validation experiment."""
    print("--- Running Experiment: Validating Criticality Mechanism ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Generate one fixed dataset
    params = CONFIG['distribution']['params']
    train_data = generate_data(params, CONFIG['n_train'], seed=42)
    test_data = generate_data(params, CONFIG['n_test'], seed=101)

    models_to_run = {
        'Unconstrained': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
        'Sub-critical': ConstrainedNSB(hidden_dim=CONFIG['nn_params']['hidden_dim'], constraint='subcritical'),
        'Critical': ConstrainedNSB(hidden_dim=CONFIG['nn_params']['hidden_dim'], constraint='critical')
    }

    for seed in tqdm(CONFIG['seeds'], desc="Running seeds"):
        for model_name, model in models_to_run.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
            
            log_likelihood = model.log_likelihood(test_data)
            eigenvalues = get_eigenvalues(model)
            spectral_radius = np.max(np.abs(eigenvalues))
            
            results.append({
                'Model': model_name,
                'Seed': seed,
                'Test Log-Likelihood': log_likelihood,
                'Spectral Radius': spectral_radius
            })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "criticality_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Figure Generation ---
def create_criticality_figure(results_df: pd.DataFrame):
    """Generates and saves the two-panel figure for the paper."""
    print("\n--- Generating Figure 3: Probing the Critical Point ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Use the results from the FIRST seed to generate a representative plot ---
    plot_seed = CONFIG['seeds'][0]
    plot_results = results_df[results_df['Seed'] == plot_seed]

    # --- Panel (a): Distributional Fits ---
    ax1 = axes[0]
    params = CONFIG['distribution']['params']
    train_data = generate_data(params, CONFIG['n_train'], seed=42)
    
    # Ground Truth
    k_max_plot = 150
    k_vals = np.arange(k_max_plot + 1)
    # true_pmf = nbinom.pmf(k_vals, n=params['n'], p=params['p'])
    true_pmf = pathological_pmf(k_vals, **params)
    ax1.plot(k_vals, true_pmf, 'o-', color=NSB_COLORS['truth'], label='Ground Truth', markersize=3, zorder=5)

    # Train and plot one representative of each model type
    models_to_plot = {
        'Unconstrained': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
        'Sub-critical': ConstrainedNSB(hidden_dim=CONFIG['nn_params']['hidden_dim'], constraint='subcritical'),
        'Critical': ConstrainedNSB(hidden_dim=CONFIG['nn_params']['hidden_dim'], constraint='critical')
    }
    
    eigenvalues_dict = {}
    for model_name, model in models_to_plot.items():
        torch.manual_seed(plot_seed)
        np.random.seed(plot_seed)
        model.fit(train_data, epochs=CONFIG['nn_params']['epochs'])
        pmf = model.predict_pmf(k_max=k_max_plot)
        eigenvalues_dict[model_name] = get_eigenvalues(model)
        
        # --- Get LL for the legend ---
        log_likelihood = plot_results[plot_results['Model'] == model_name]['Test Log-Likelihood'].iloc[0]
        legend_label = f"{model_name} (LL: {log_likelihood:.2f})"
        
        color = NSB_COLORS['nsb'] if 'Unconstrained' in model_name else (NSB_COLORS['nsb_subcritical'] if 'Sub' in model_name else 'orange')
        ax1.plot(k_vals, pmf, '--', color=color, label=legend_label)

    ax1.set_title("(a) Distributional Fits", weight='bold')
    ax1.set_xlabel("Offspring Count (k)")
    ax1.set_ylabel("Probability (log scale)")
    ax1.set_yscale('log')
    ax1.legend()

    # --- Panel (b): Spectral Analysis (no changes here) ---
    ax2 = axes[1]
    unit_circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5, label='Unit Circle')
    ax2.add_patch(unit_circle)

    markers = {'Unconstrained': 'o', 'Sub-critical': 's', 'Critical': '^'}
    colors = {'Unconstrained': NSB_COLORS['nsb'], 'Sub-critical': NSB_COLORS['nsb_subcritical'], 'Critical': 'orange'}

    for model_name, eigenvalues in eigenvalues_dict.items():
        ax2.scatter(eigenvalues.real, eigenvalues.imag, 
                    marker=markers[model_name], 
                    color=colors[model_name], 
                    label=model_name, 
                    alpha=0.7, s=50)

    ax2.set_title("(b) Spectral Analysis of Learned Weights", weight='bold')
    ax2.set_xlabel("Real Part")
    ax2.set_ylabel("Imaginary Part")
    ax2.axhline(0, color='grey', lw=0.5)
    ax2.axvline(0, color='grey', lw=0.5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    ax2.grid(True)

    save_figure(fig, "criticality_validation")


if __name__ == "__main__":
    results_df = run_experiment()
    create_criticality_figure(results_df)

"""
This script trains and evaluates the NSB model and all baselines on the
processed `outbreaktrees` dataset of SARS and MERS transmission events.

The script generates:
1.  A CSV file with the raw results (`results/real_world_results.csv`).
2.  A LaTeX table for the main benchmark results (Table 2).
3.  A LaTeX table for the ablation study (Table 3).
4.  A three-panel figure analyzing the results, including a rank-frequency
    plot with a spectral analysis inset (Figure 5).
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import time

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
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'test_size': 0.2,
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
        'k_max': 150 # For SoftmaxNN
    }
}

# --- Helper Functions ---
def load_and_split_data(path: Path, test_size: float, seed: int):
    """Loads the processed count data and splits it into train/test sets."""
    df = pd.read_csv(path)
    counts = df['offspring_count'].values
    return train_test_split(counts, test_size=test_size, random_state=seed)

def get_eigenvalues(model: NSB) -> np.ndarray:
    """Calculates the eigenvalues of the recurrent weight matrix W_h."""
    weight_matrix = model.cell.fc_h.weight.data.cpu().numpy()
    return np.linalg.eigvals(weight_matrix)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the KL divergence D_KL(P || Q) for discrete distributions."""
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))

# --- Main Experiment Loop ---
def run_experiment():
    """Runs the full real-world case study experiment."""
    print("--- Running Experiment: Real-World Case Study ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results = []

    for seed in tqdm(CONFIG['seeds'], desc="Running seeds"):
        train_data, test_data = load_and_split_data(CONFIG['data_path'], CONFIG['test_size'], seed)
        
        fair_hidden_dim = 27
        models = {
            'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
            # 'NSB (no hidden layer)': NSB(hidden_dim=0),
            'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim),
            'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=CONFIG['nn_params']['hidden_dim']),
            'Negative Binomial': NegativeBinomialMLE(),
            'Poisson': PoissonMLE()
        }

        # --- Define the tail for KL divergence calculation ---
        tail_start_idx = int(np.median(train_data)) + 1
        k_max_eval = test_data.max()
        
        # Empirical tail distribution from test data
        test_tail_counts = test_data[test_data >= tail_start_idx]
        if len(test_tail_counts) > 0:
            empirical_tail_pmf = np.bincount(test_tail_counts, minlength=k_max_eval + 1)[tail_start_idx:]
            empirical_tail_pmf = empirical_tail_pmf / empirical_tail_pmf.sum()
        else:
            empirical_tail_pmf = None

        for model_name, model in models.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                if hasattr(model, 'fit'):
                    if isinstance(model, (NSB, SoftmaxNN)):
                        model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
                    else:
                        model.fit(train_data)
                
                start_time = time.time()
                log_likelihood = model.log_likelihood(test_data)
                end_time = time.time()
                inference_time = (end_time - start_time) / len(test_data) * 1000

                # Calculate tail KL divergence
                tail_kl = np.nan
                if empirical_tail_pmf is not None:
                    if isinstance(model, SoftmaxNN):
                        pmf_full = model.predict_pmf()
                        learned_pmf = np.zeros(k_max_eval + 1)
                        len_to_copy = min(len(pmf_full), len(learned_pmf))
                        learned_pmf[:len_to_copy] = pmf_full[:len_to_copy]
                    elif hasattr(model, 'predict_pmf'): # This will now only be the NSB
                        learned_pmf = model.predict_pmf(k_max=k_max_eval)
                    else: # For Poisson and NegativeBinomial
                        learned_pmf = model.pmf(np.arange(k_max_eval + 1))
                    
                    learned_tail_pmf = learned_pmf[tail_start_idx:]
                    if learned_tail_pmf.sum() > 1e-9: # Avoid division by zero
                        learned_tail_pmf = learned_tail_pmf / learned_tail_pmf.sum()
                        tail_kl = kl_divergence(empirical_tail_pmf, learned_tail_pmf)

                if isinstance(model, NSB):
                    num_params = count_parameters(model.cell) + model.h0.numel() if model.hidden_dim > 0 else 2
                elif isinstance(model, SoftmaxNN):
                    num_params = count_parameters(model.model)
                else:
                    num_params = 2 if isinstance(model, NegativeBinomialMLE) else 1

            except (ValueError, RuntimeError) as e:
                log_likelihood, inference_time, num_params, tail_kl = -np.inf, -1, -1, np.inf

            results.append({
                'Model': model_name,
                'Seed': seed,
                'Test Log-Likelihood': log_likelihood,
                'Num. Params': num_params,
                'Inference Time (ms)': inference_time,
                'Tail KL Divergence': tail_kl
            })

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "real_world_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Table and Figure Generation ---
def create_tables(results_df: pd.DataFrame):
    """Generates and saves the LaTeX tables for the paper."""
    
    # --- Main Benchmark Results ---
    main_models = ['NSB (Ours)', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    main_df = results_df[results_df['Model'].isin(main_models)]
    
    summary = main_df.groupby('Model').agg({
        'Test Log-Likelihood': ['mean', 'std'],
        'Tail KL Divergence': ['mean', 'std'],
        'Num. Params': 'first',
        'Inference Time (ms)': 'mean'
    }).reindex(main_models)

    best_ll = summary[('Test Log-Likelihood', 'mean')].max()
    best_kl = summary[('Tail KL Divergence', 'mean')].min()

    latex_str = "\\begin{tabular}{lcccc}\n"
    latex_str += "\\toprule\n"
    latex_str += "Model & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Num. Params & Time (ms) \\\\\n"
    latex_str += "\\midrule\n"
    for model_name, row in summary.iterrows():
        ll_mean, ll_std = row[('Test Log-Likelihood', 'mean')], row[('Test Log-Likelihood', 'std')]
        kl_mean, kl_std = row[('Tail KL Divergence', 'mean')], row[('Tail KL Divergence', 'std')]
        num_params = int(row[('Num. Params', 'first')])
        inf_time = row[('Inference Time (ms)', 'mean')]
        
        ll_str = f"${ll_mean:.2f} \\pm {ll_std:.2f}$"
        if ll_mean == best_ll: ll_str = f"\\textbf{{{ll_str}}}"
        
        kl_str = f"${kl_mean:.2f} \\pm {kl_std:.2f}$"
        if kl_mean == best_kl: kl_str = f"\\textbf{{{kl_str}}}"

        latex_str += f"{model_name} & {ll_str} & {kl_str} & {num_params:,} & {inf_time:.2f} \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}"
    
    table_path = CONFIG['output_dir_results'] / "real_world_benchmark_table.tex"
    with open(table_path, 'w') as f: f.write(latex_str)
    print(f"\nLaTeX Table 2 saved to '{table_path}'")

def create_real_world_figure():
    """Generates the three-panel figure for the real-world analysis."""
    print("\n--- Generating Figure: Real-World Distribution Analysis ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    plot_seed = CONFIG['seeds'][0]
    train_data, test_data = load_and_split_data(CONFIG['data_path'], CONFIG['test_size'], plot_seed)
    
    # --- Panel (a): Overall Fit ---
    ax1 = axes[0]
    max_val_hist = int(np.percentile(test_data, 98))
    bins = np.arange(max_val_hist + 2) - 0.5
    ax1.hist(test_data, bins=bins, density=True, color='gray', alpha=0.3, label='Empirical Data')
    
    models_to_plot = {
        'NSB (Ours)': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
        'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max']),
        'Negative Binomial': NegativeBinomialMLE(),
        # 'Poisson': PoissonMLE(),
    }
    
    k_vals_plot = np.arange(max_val_hist + 1)
    for model_name, model in models_to_plot.items():
        torch.manual_seed(plot_seed); np.random.seed(plot_seed)
        if isinstance(model, (NSB, SoftmaxNN)):
            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
        else:
            model.fit(train_data)
        if isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(max_val_hist + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif hasattr(model, 'predict_pmf'):
            pmf = model.predict_pmf(k_max=max_val_hist)
        else:
            pmf = model.pmf(k_vals_plot)
        color_key = model_name.lower().replace(' (ours)','').replace(' ','_')
        ax1.plot(k_vals_plot, pmf, label=model_name, color=NSB_COLORS[color_key])
        
    ax1.set_title("(a) Overall Distributional Fit", weight='bold')
    ax1.set_xlabel("Offspring Count (k)"); ax1.set_ylabel("Probability (log scale)"); ax1.legend()
    ax1.set_yscale('log') 

    # --- Panel (b): Log-Log Tail Analysis ---
    ax2 = axes[1]
    counts, freqs = np.unique(test_data, return_counts=True)
    probs = freqs / len(test_data)
    ax2.plot(counts, probs, 'o', color=NSB_COLORS['truth'], label='Empirical Data', markersize=4)
    
    k_max_tail = train_data.max()
    for model_name, model in models_to_plot.items():
        if isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif hasattr(model, 'predict_pmf'):
            pmf = model.predict_pmf(k_max=k_max_tail)
        else:
            pmf = model.pmf(np.arange(k_max_tail + 1))
        color_key = model_name.lower().replace(' (ours)','').replace(' ','_')
        ax2.plot(np.arange(k_max_tail + 1), pmf, '--', label=model_name, color=NSB_COLORS[color_key])
    
    ax2.set_title("(b) Log-Log Analysis of the Tail", weight='bold')
    ax2.set_xlabel("Offspring Count (k) (log scale)"); ax2.set_ylabel("Probability (log scale)")
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.legend()

    # --- Panel (c): Rank-Frequency Analysis ---
    ax3 = axes[2]
    sorted_probs = np.sort(probs)[::-1]
    ranks = np.arange(1, len(sorted_probs) + 1)
    ax3.plot(ranks, sorted_probs, 'o', color=NSB_COLORS['truth'], label='Empirical Data', markersize=4)
    
    for model_name, model in models_to_plot.items():
        if isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif hasattr(model, 'predict_pmf'):
            pmf = model.predict_pmf(k_max=k_max_tail)
        else:
            pmf = model.pmf(np.arange(k_max_tail + 1))
        
        pmf_sorted = np.sort(pmf[pmf > 1e-9])[::-1]
        ranks_model = np.arange(1, len(pmf_sorted) + 1)
        color_key = model_name.lower().replace(' (ours)','').replace(' ','_')
        ax3.plot(ranks_model, pmf_sorted, '--', label=model_name, color=NSB_COLORS[color_key])

    ax3.set_title("(c) Rank-Frequency Analysis (Zipf Plot)", weight='bold')
    ax3.set_xlabel("Rank (log scale)"); ax3.set_ylabel("Probability (log scale)")
    ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.legend()

    # --- Inset for Spectral Analysis ---
    from matplotlib.patches import Circle
    nsb_model = models_to_plot['NSB (Ours)']
    axins = ax3.inset_axes([0.02, 0.02, 0.45, 0.45])
    unit_circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
    axins.add_patch(unit_circle)
    eigenvalues = get_eigenvalues(nsb_model)
    axins.scatter(eigenvalues.real, eigenvalues.imag, marker='o', color=NSB_COLORS['nsb'], alpha=0.7, s=20)
    axins.set_title("NSB's Spectral Radius", fontsize=10)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_aspect('equal', adjustable='box')

    save_figure(fig, "real_world_analysis")

if __name__ == "__main__":
    results_df = run_experiment()
    create_tables(results_df)
    create_real_world_figure()
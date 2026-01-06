"""
This script generates plots and tables from the real-world experiment results.

The script generates:
1.  A LaTeX table for the main benchmark results (Table 2).
2.  A three-panel figure analyzing the results, including a rank-frequency
    plot with a spectral analysis inset (Figure 5).
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from nsb.gru_model import NSBGRU
from nsb.lstm_model import NSBLSTM
from nsb.attention_model import NSBAttention
from nsb.poisson import PoissonMLE
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.softmax_nn import SoftmaxNN
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

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

# --- Table and Figure Generation ---
def create_tables(results_df: pd.DataFrame):
    """Generates and saves the LaTeX tables for the paper."""
    
    # Map old model names to new ones for backward compatibility
    model_name_mapping = {
        'NSB (Ours)': 'NSB',
        'NSBGRU': 'NSB-GRU',
        'NSBLSTM': 'NSB-LSTM',
        'NSBAttention': 'NSB-Attention'
    }
    results_df = results_df.copy()
    results_df['Model'] = results_df['Model'].replace(model_name_mapping)
    
    # --- Main Benchmark Results ---
    # Order: NSB models first, then SoftmaxNN, then statistical baselines
    main_models = ['NSB', 'NSB-GRU', 'NSB-LSTM', 'NSB-Attention', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    main_df = results_df[results_df['Model'].isin(main_models)]
    
    summary = main_df.groupby('Model').agg({
        'Test Log-Likelihood': ['mean', 'std'],
        'Tail KL Divergence': ['mean', 'std'],
        'Num. Params': 'first'
    }).reindex(main_models)

    best_ll = summary[('Test Log-Likelihood', 'mean')].max()
    best_kl = summary[('Tail KL Divergence', 'mean')].min()

    latex_str = "\\begin{tabular}{lccc}\n"
    latex_str += "\\toprule\n"
    latex_str += "Model & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Num. Params \\\\\n"
    latex_str += "\\midrule\n"
    for model_name, row in summary.iterrows():
        ll_mean, ll_std = row[('Test Log-Likelihood', 'mean')], row[('Test Log-Likelihood', 'std')]
        kl_mean, kl_std = row[('Tail KL Divergence', 'mean')], row[('Tail KL Divergence', 'std')]
        num_params = int(row[('Num. Params', 'first')])
        
        ll_str = f"${ll_mean:.2f} \\pm {ll_std:.2f}$"
        if ll_mean == best_ll: ll_str = f"\\textbf{{{ll_str}}}"
        
        kl_str = f"${kl_mean:.2f} \\pm {kl_std:.2f}$"
        if kl_mean == best_kl: kl_str = f"\\textbf{{{kl_str}}}"

        latex_str += f"{model_name} & {ll_str} & {kl_str} & {num_params:,} \\\\\n"
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
    
    # Map model names to color keys
    color_map = {
        'NSB': 'nsb',
        'NSB-GRU': 'nsb_gru',
        'NSB-LSTM': 'nsb_lstm',
        'NSB-Attention': 'nsb_attention',
        'Softmax NN (Fair)': 'softmax_nn_(fair)',
        'Softmax NN': 'softmax_nn',
        'Negative Binomial': 'negative_binomial',
        'Poisson': 'poisson'
    }
    
    # --- Panel (a): Overall Fit ---
    ax1 = axes[0]
    max_val_hist = int(np.percentile(test_data, 98))
    bins = np.arange(max_val_hist + 2) - 0.5
    ax1.hist(test_data, bins=bins, density=True, color='gray', alpha=0.3, label='Empirical Data')
    
    # Hidden dimensions from count_parameters.py for fair parameter-matched comparison
    # NSB: hidden_dim=64 (4,289 params)
    # NSBGRU: hidden_dim=27 (4,591 params)
    # NSBLSTM: hidden_dim=23 (4,486 params)
    # NSBAttention: hidden_dim=19, max_k=150 (4,409 params)
    # SoftmaxNN (Fair): hidden_dim=28 (4,435 params)
    fair_hidden_dim = 28
    models_to_plot = {
        'NSB': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
        'NSB-GRU': NSBGRU(hidden_dim=27),
        'NSB-LSTM': NSBLSTM(hidden_dim=23),
        'NSB-Attention': NSBAttention(hidden_dim=19, num_heads=1, max_k=CONFIG['nn_params']['k_max']),
        'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim),
        'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max']),
        'Negative Binomial': NegativeBinomialMLE(),
        # 'Poisson': PoissonMLE(),
    }
    
    k_vals_plot = np.arange(max_val_hist + 1)
    for model_name, model in models_to_plot.items():
        torch.manual_seed(plot_seed); np.random.seed(plot_seed)
        # Check NSBAttention first since it inherits from NSB
        if isinstance(model, NSBAttention):
            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
        elif isinstance(model, (NSB, NSBGRU, NSBLSTM, SoftmaxNN)):
            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
        else:
            model.fit(train_data)
        
        # Check NSBAttention first since it inherits from NSB
        if isinstance(model, NSBAttention):
            k_max_for_pmf = min(max_val_hist, model.max_k)
            pmf_partial = model.predict_pmf(k_max=k_max_for_pmf)
            pmf = np.zeros(max_val_hist + 1)
            len_to_copy = min(len(pmf_partial), len(pmf))
            pmf[:len_to_copy] = pmf_partial[:len_to_copy]
        elif isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(max_val_hist + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
            pmf = model.predict_pmf(k_max=max_val_hist)
        else:
            pmf = model.pmf(k_vals_plot)
        
        color_key = color_map.get(model_name, model_name.lower().replace(' (ours)','').replace(' ','_'))
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
        # Check NSBAttention first since it inherits from NSB
        if isinstance(model, NSBAttention):
            k_max_for_pmf = min(k_max_tail, model.max_k)
            pmf_partial = model.predict_pmf(k_max=k_max_for_pmf)
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_partial), len(pmf))
            pmf[:len_to_copy] = pmf_partial[:len_to_copy]
        elif isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
            pmf = model.predict_pmf(k_max=k_max_tail)
        else:
            pmf = model.pmf(np.arange(k_max_tail + 1))
        
        color_key = color_map.get(model_name, model_name.lower().replace(' (ours)','').replace(' ','_'))
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
        # Check NSBAttention first since it inherits from NSB
        if isinstance(model, NSBAttention):
            k_max_for_pmf = min(k_max_tail, model.max_k)
            pmf_partial = model.predict_pmf(k_max=k_max_for_pmf)
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_partial), len(pmf))
            pmf[:len_to_copy] = pmf_partial[:len_to_copy]
        elif isinstance(model, SoftmaxNN):
            pmf_full = model.predict_pmf()
            pmf = np.zeros(k_max_tail + 1)
            len_to_copy = min(len(pmf_full), len(pmf))
            pmf[:len_to_copy] = pmf_full[:len_to_copy]
        elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
            pmf = model.predict_pmf(k_max=k_max_tail)
        else:
            pmf = model.pmf(np.arange(k_max_tail + 1))
        
        pmf_sorted = np.sort(pmf[pmf > 1e-9])[::-1]
        ranks_model = np.arange(1, len(pmf_sorted) + 1)
        color_key = color_map.get(model_name, model_name.lower().replace(' (ours)','').replace(' ','_'))
        ax3.plot(ranks_model, pmf_sorted, '--', label=model_name, color=NSB_COLORS[color_key])

    ax3.set_title("(c) Rank-Frequency Analysis (Zipf Plot)", weight='bold')
    ax3.set_xlabel("Rank (log scale)"); ax3.set_ylabel("Probability (log scale)")
    ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.legend(loc='upper right')

    # --- Inset for Spectral Analysis ---
    nsb_model = models_to_plot['NSB']
    axins = ax3.inset_axes([0.08, 0.06, 0.45, 0.45])
    
    # Draw unit circle
    unit_circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
    axins.add_patch(unit_circle)
    
    # Draw axes through origin
    axins.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    axins.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Get eigenvalues and plot them
    eigenvalues = get_eigenvalues(nsb_model)
    axins.scatter(eigenvalues.real, eigenvalues.imag, marker='o', color=NSB_COLORS['nsb'], alpha=0.7, s=20)
    
    # Find eigenvalues outside the unit circle and annotate them
    for i, eig in enumerate(eigenvalues):
        magnitude = np.abs(eig)
        if magnitude > 1.0:
            # Annotate with a small offset to avoid overlapping with the point
            axins.annotate(f'{magnitude:.2f}', 
                          xy=(eig.real, eig.imag),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, color='red', weight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
    
    # Set axis labels
    axins.set_xlabel('Re', fontsize=9, labelpad=2)
    axins.set_ylabel('Im', fontsize=9, labelpad=2)
    
    # Set ticks to show a few key values
    axins.set_xticks([-1, 0, 1])
    axins.set_yticks([-1, 0, 1])
    axins.tick_params(labelsize=7)
    
    axins.set_title("NSB's Spectral Radius", fontsize=10)
    axins.set_aspect('equal', adjustable='box')

    save_figure(fig, "real_world_analysis")

if __name__ == "__main__":
    # Load results from CSV
    results_path = CONFIG['output_dir_results'] / "real_world_results.csv"
    if not results_path.exists():
        print(f"Error: Results file not found at '{results_path}'")
        print("Please run 'exp_outbreaktrees.py' first to generate the results.")
        exit(1)
    
    results_df = pd.read_csv(results_path)
    create_tables(results_df)
    create_real_world_figure()


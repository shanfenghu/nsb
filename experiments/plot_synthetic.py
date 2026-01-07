"""
This script generates plots and tables from the experimental results.

The script generates:
1.  A LaTeX-formatted performance table.
2.  A figure showing the learning dynamics of Test Log-Likelihood and
    Tail KL Divergence for each model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# --- Configuration ---
CONFIG = {
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'nn_params': {
        'epochs': 50,
    },
    'distributions': {
        'Poisson': {'type': 'poisson', 'params': {'mu': 3}},
        'Negative Binomial': {'type': 'neg_binomial', 'params': {'n': 2, 'p': 0.1}},
        'Zero-Inflated NB': {'type': 'zinb', 'params': {'n': 2, 'p': 0.1, 'omega': 0.5}}
    }
}

# --- Table and Figure Generation ---
def create_results_table(results_df: pd.DataFrame):
    """Generates and saves the LaTeX table."""
    # Map old model names from CSV to new display names
    model_name_mapping = {
        'NSB (Ours)': 'NSB',
        'NSBGRU': 'NSB-GRU',
        'NSBLSTM': 'NSB-LSTM',
        'NSBAttention': 'NSB-Attention'
    }
    # Apply mapping to results_df
    results_df = results_df.copy()
    results_df['Model'] = results_df['Model'].replace(model_name_mapping)
    
    final_epoch_results = results_df[results_df['Epoch'] == CONFIG['nn_params']['epochs']]
    summary = final_epoch_results.groupby(['Model', 'Distribution']).agg(
        ll_mean=('Test Log-Likelihood', 'mean'), ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'), kl_std=('Tail KL Divergence', 'std')
    ).unstack()

    model_order = ['NSB', 'NSB-GRU', 'NSB-LSTM', 'NSB-Attention', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    dist_order = ['Poisson', 'Negative Binomial', 'Zero-Inflated NB']
    summary = summary.reindex(model_order)
    summary = summary.reindex(columns=dist_order, level='Distribution')

    latex_str = "\\begin{tabular}{l" + "cc" * len(dist_order) + "}\n\\toprule\n"
    latex_str += " & \\multicolumn{2}{c}{Poisson} & \\multicolumn{2}{c}{Negative Binomial} & \\multicolumn{2}{c}{Zero-Inflated NB} \\\\\n"
    latex_str += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \n"
    latex_str += "Model & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Test LL $\\uparrow$ & Tail KL $\\downarrow$ \\\\\n\\midrule\n"
    # Note: Requires xcolor package in LaTeX preamble: \usepackage{xcolor}

    for model_name, row in summary.iterrows():
        latex_str += f"{model_name}"
        for dist_name in dist_order:
            ll_mean, ll_std = row[('ll_mean', dist_name)], row[('ll_std', dist_name)]
            kl_mean, kl_std = row[('kl_mean', dist_name)], row[('kl_std', dist_name)]
            
            ll_str = f"${ll_mean:.3f} \\pm {ll_std:.3f}$"
            kl_str = f"${kl_mean:.3f} \\pm {kl_std:.3f}$"
            
            # Highlight best results in bold and blue
            if ll_mean == summary['ll_mean'][dist_name].max(): 
                ll_str = f"\\textcolor{{blue}}{{\\textbf{{{ll_str}}}}}"
            if kl_mean == summary['kl_mean'][dist_name].min(): 
                kl_str = f"\\textcolor{{blue}}{{\\textbf{{{kl_str}}}}}"
            
            latex_str += f" & {ll_str} & {kl_str}"
        latex_str += " \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}"

    table_path = CONFIG['output_dir_results'] / "synthetic_dynamics.tex"
    with open(table_path, 'w') as f: f.write(latex_str)
    print(f"\nLaTeX Table saved to '{table_path}'")

def create_dynamics_figure(results_df: pd.DataFrame):
    """Generates the learning dynamics figure."""
    print("\n--- Generating Figure: Learning Dynamics on Noisy Data ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    # Map old model names from CSV to new display names
    model_name_mapping = {
        'NSB (Ours)': 'NSB',
        'NSBGRU': 'NSB-GRU',
        'NSBLSTM': 'NSB-LSTM',
        'NSBAttention': 'NSB-Attention'
    }
    # Apply mapping to results_df
    results_df = results_df.copy()
    results_df['Model'] = results_df['Model'].replace(model_name_mapping)

    summary = results_df.groupby(['Distribution', 'Model', 'Epoch']).agg(
        ll_mean=('Test Log-Likelihood', 'mean'), ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'), kl_std=('Tail KL Divergence', 'std')
    ).reset_index()

    model_order = ['NSB', 'NSB-GRU', 'NSB-LSTM', 'NSB-Attention', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    colors = {
        'NSB': NSB_COLORS['nsb'],
        'NSB-GRU': NSB_COLORS['nsb_gru'],
        'NSB-LSTM': NSB_COLORS['nsb_lstm'],
        'NSB-Attention': NSB_COLORS['nsb_attention'],
        'Softmax NN (Fair)': NSB_COLORS['softmax_nn_(fair)'],
        'Softmax NN': NSB_COLORS['softmax_nn'],
        'Negative Binomial': NSB_COLORS['negative_binomial'],
        'Poisson': NSB_COLORS['poisson']
    }
    linestyles = {'Test Log-Likelihood': '-', 'Tail KL Divergence': '--'}

    # Collect handles and labels for legend in the exact model order
    # We'll create proxy artists to ensure all models appear in legend even if not in all subplots
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    # NSB models to bold in legend
    nsb_models = ['NSB', 'NSB-GRU', 'NSB-LSTM', 'NSB-Attention']
    for model_name in model_order:
        # Create a proxy line for the legend with the correct color and linestyle
        proxy_line = Line2D([0], [0], color=colors.get(model_name, '#000000'), 
                           linestyle=linestyles['Test Log-Likelihood'], linewidth=2.5)
        legend_handles.append(proxy_line)
        legend_labels.append(model_name)

    for i, dist_name in enumerate(CONFIG['distributions'].keys()):
        ax = axes[i]
        ax2 = ax.twinx() # Create a second y-axis

        for model_name in model_order:
            model_data = summary[(summary['Distribution'] == dist_name) & (summary['Model'] == model_name)]
            
            if len(model_data) == 0:
                continue  # Skip if no data for this model
            
            # Plot Test LL on the left axis
            ax.plot(model_data['Epoch'], model_data['ll_mean'], color=colors.get(model_name, '#000000'), linestyle=linestyles['Test Log-Likelihood'], label=model_name)
            ax.fill_between(model_data['Epoch'], model_data['ll_mean'] - model_data['ll_std'], model_data['ll_mean'] + model_data['ll_std'], color=colors.get(model_name, '#000000'), alpha=0.1)
            
            # Plot Tail KL on the right axis
            ax2.plot(model_data['Epoch'], model_data['kl_mean'], color=colors.get(model_name, '#000000'), linestyle=linestyles['Tail KL Divergence'])
            ax2.fill_between(model_data['Epoch'], model_data['kl_mean'] - model_data['kl_std'], model_data['kl_mean'] + model_data['kl_std'], color=colors.get(model_name, '#000000'), alpha=0.1)

        ax.set_title(f"({chr(97 + i)}) {dist_name}", weight='bold')
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Test Log-Likelihood (Higher is better)", weight='bold', color='black')
        ax2.set_ylabel("Tail KL Divergence (Lower is better)", weight='bold', color='gray')
        ax2.set_yscale('log')
    
    # Create a unified legend with all models in one row:
    # NSB, NSB-GRU, NSB-LSTM, NSB-Attention, Softmax NN (Fair), Softmax NN, Negative Binomial, Poisson
    legend = fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(model_order), bbox_to_anchor=(0.5, -0.08))
    
    # Bold the four NSB models in the legend
    nsb_models = ['NSB', 'NSB-GRU', 'NSB-LSTM', 'NSB-Attention']
    for text in legend.get_texts():
        if text.get_text() in nsb_models:
            text.set_weight('bold')

    save_figure(fig, "synthetic_dynamics")

if __name__ == "__main__":
    # Load results from CSV
    results_path = CONFIG['output_dir_results'] / "synthetic_dynamics.csv"
    if not results_path.exists():
        print(f"Error: Results file not found at '{results_path}'")
        print("Please run 'python experiments/exp_synthetic.py' first to generate the results.")
        exit(1)
    
    results_df = pd.read_csv(results_path)
    print(f"Loaded results from '{results_path}'")
    
    create_results_table(results_df)
    create_dynamics_figure(results_df)


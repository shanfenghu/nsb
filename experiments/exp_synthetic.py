"""
This script provides a deep, dynamic view of how different models learn from
noisy, imperfect training data. It tracks the performance of each model,
epoch by epoch, on two challenging, heavy-tailed synthetic distributions.

The script generates:
1.  A CSV file with the full learning curve results (`results/synthetic_dynamics_results.csv`).
2.  A compact, revised LaTeX-formatted performance table for the paper.
3.  A two-panel figure showing the learning dynamics of Test Log-Likelihood and
    Tail KL Divergence for each model.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import poisson, nbinom
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
CONFIG = {
    'seeds': list(range(5)),
    'n_train': 5000,
    'n_test': 1000,
    'noise_fraction': 0.00,
    'noise_range': (0, 20),
    'output_dir_results': Path("results"),
    'output_dir_figures': Path("figures"),
    'distributions': {
        'Poisson': {'type': 'poisson', 'params': {'mu': 3}},
        'Negative Binomial': {'type': 'neg_binomial', 'params': {'n': 2, 'p': 0.1}},
        'Zero-Inflated NB': {'type': 'zinb', 'params': {'n': 2, 'p': 0.1, 'omega': 0.5}}
    },
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
        'k_max': 150
    },
    'k_max_eval': 400
}

# --- Helper Functions ---
def zinb_pmf(k, n, p, omega):
    nb_pmf_vals = nbinom.pmf(k, n=n, p=p)
    pmf = (1 - omega) * nb_pmf_vals
    if isinstance(k, (int, float)) and k == 0: pmf = omega + (1 - omega) * nb_pmf_vals
    elif isinstance(k, np.ndarray): pmf[k == 0] = omega + (1 - omega) * nb_pmf_vals[k == 0]
    return pmf

def generate_data(dist_info: dict, n_samples: int, seed: int, add_noise: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = dist_info['params']
    if dist_info['type'] == 'poisson':
        counts = rng.poisson(params['mu'], n_samples)
    elif dist_info['type'] == 'neg_binomial':
        counts = rng.negative_binomial(params['n'], params['p'], n_samples)
    elif dist_info['type'] == 'zinb':
        counts = rng.negative_binomial(params['n'], params['p'], n_samples)
        mask = rng.binomial(1, params['omega'], n_samples).astype(bool)
        counts[mask] = 0
    else: raise ValueError(f"Unknown distribution type: {dist_info['type']}")
    if add_noise:
        n_noise = int(n_samples * CONFIG['noise_fraction'])
        noise_indices = rng.choice(n_samples, n_noise, replace=False)
        noise_values = rng.integers(CONFIG['noise_range'][0], CONFIG['noise_range'][1] + 1, size=n_noise)
        counts[noise_indices] = noise_values
    return counts

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-10; q = q + 1e-10
    return np.sum(p * np.log(p / q))

# --- Main Experiment Loop ---
def run_experiments():
    print("--- Running Experiment: Learning Dynamics ---")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    results = []
    
    for dist_name, dist_info in CONFIG['distributions'].items():
        print(f"\nProcessing Distribution: {dist_name}")
        train_data = generate_data(dist_info, CONFIG['n_train'], seed=42, add_noise=True)
        test_data = generate_data(dist_info, CONFIG['n_test'], seed=101, add_noise=False)
        
        median_val = int(np.median(generate_data(dist_info, 10000, 42)))
        tail_start_idx = median_val + 1
        k_vals_eval = np.arange(CONFIG['k_max_eval'] + 1)
        if dist_info['type'] == 'poisson': true_pmf = poisson.pmf(k_vals_eval, **dist_info['params'])
        elif dist_info['type'] == 'neg_binomial': true_pmf = nbinom.pmf(k_vals_eval, **dist_info['params'])
        else: true_pmf = zinb_pmf(k_vals_eval, **dist_info['params'])
        true_tail_pmf = true_pmf[tail_start_idx:]; true_tail_pmf /= true_tail_pmf.sum()

        for seed in tqdm(CONFIG['seeds'], desc=f"  Running seeds for {dist_name}"):
            torch.manual_seed(seed); np.random.seed(seed)
            # train_data = generate_data(dist_info, CONFIG['n_train'], seed=seed, add_noise=True)
            # test_data = generate_data(dist_info, CONFIG['n_test'], seed=seed, add_noise=False)

            # Evaluate non-trainable models once
            for model_class, model_name in [(PoissonMLE, 'Poisson'), (NegativeBinomialMLE, 'Negative Binomial')]:
                model = model_class()
                try:
                    model.fit(train_data)
                    ll = model.log_likelihood(test_data)
                    pmf = model.pmf(k_vals_eval)
                    tail_pmf = pmf[tail_start_idx:]; tail_pmf /= tail_pmf.sum()
                    kl = kl_divergence(true_tail_pmf, tail_pmf)
                except (ValueError, RuntimeError):
                    ll, kl = -np.inf, np.inf
                for epoch in range(CONFIG['nn_params']['epochs'] + 1):
                    results.append({'Distribution': dist_name, 'Model': model_name, 'Seed': seed, 'Epoch': epoch,
                                    'Test Log-Likelihood': ll, 'Tail KL Divergence': kl})

            # Train and evaluate neural models epoch-wise
            fair_hidden_dim = 27
            nn_models = {'NSB (Ours)': NSB(), 'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim),
                         'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'])}
            
            for model_name, model in nn_models.items():
                optimizer = torch.optim.Adam(list(model.cell.parameters()) + [model.h0] if isinstance(model, NSB) else model.model.parameters(), lr=CONFIG['nn_params']['lr'])
                loader = DataLoader(TensorDataset(torch.from_numpy(train_data).long()), batch_size=CONFIG['nn_params']['batch_size'], shuffle=True)
                
                for epoch in range(CONFIG['nn_params']['epochs'] + 1):
                    if epoch > 0: # Training step
                        model.cell.train() if isinstance(model, NSB) else model.model.train()
                        for batch_counts in loader:
                            optimizer.zero_grad()
                            if isinstance(model, NSB):
                                log_probs = model._compute_log_probs(batch_counts[0].to(model.device))
                                loss = -log_probs.mean()
                            else: # SoftmaxNN
                                dummy_input = torch.zeros(len(batch_counts[0]), 1)
                                logits = model.model(dummy_input)
                                loss = torch.nn.functional.cross_entropy(logits, batch_counts[0])
                            loss.backward(); optimizer.step()
                    
                    # Evaluation step (at epoch 0 and after each training epoch)
                    if isinstance(model, SoftmaxNN):
                        # For SoftmaxNN, we must manually compute the PMF from the
                        # current model state before evaluation.
                        model.model.eval()
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, 1, device=model.model.network[0].weight.device)
                            logits = model.model(dummy_input)
                            model.pmf_ = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

                    ll = model.log_likelihood(test_data)

                    if isinstance(model, SoftmaxNN):
                        pmf_full = model.predict_pmf(); learned_pmf = np.zeros(CONFIG['k_max_eval'] + 1)
                        len_to_copy = min(len(pmf_full), len(learned_pmf)); learned_pmf[:len_to_copy] = pmf_full[:len_to_copy]
                    else: learned_pmf = model.predict_pmf(k_max=CONFIG['k_max_eval'])
                    tail_pmf = learned_pmf[tail_start_idx:]
                    if tail_pmf.sum() > 1e-9: tail_pmf /= tail_pmf.sum(); kl = kl_divergence(true_tail_pmf, tail_pmf)
                    else: kl = np.inf
                    
                    results.append({'Distribution': dist_name, 'Model': model_name, 'Seed': seed, 'Epoch': epoch,
                                    'Test Log-Likelihood': ll, 'Tail KL Divergence': kl})

    results_df = pd.DataFrame(results)
    output_path = CONFIG['output_dir_results'] / "synthetic_dynamics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to '{output_path}'")
    return results_df

# --- Table and Figure Generation ---
def create_results_table(results_df: pd.DataFrame):
    """Generates and saves the revised LaTeX table."""
    final_epoch_results = results_df[results_df['Epoch'] == CONFIG['nn_params']['epochs']]
    summary = final_epoch_results.groupby(['Model', 'Distribution']).agg(
        ll_mean=('Test Log-Likelihood', 'mean'), ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'), kl_std=('Tail KL Divergence', 'std')
    ).unstack()

    model_order = ['NSB (Ours)', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    dist_order = ['Poisson', 'Negative Binomial', 'Zero-Inflated NB']
    summary = summary.reindex(model_order)
    summary = summary.reindex(columns=dist_order, level='Distribution')

    latex_str = "\\begin{tabular}{l" + "cc" * len(dist_order) + "}\n\\toprule\n"
    latex_str += " & \\multicolumn{2}{c}{Poisson} & \\multicolumn{2}{c}{Negative Binomial} & \\multicolumn{2}{c}{Zero-Inflated NB} \\\\\n"
    latex_str += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \n"
    latex_str += "Model & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Test LL $\\uparrow$ & Tail KL $\\downarrow$ & Test LL $\\uparrow$ & Tail KL $\\downarrow$ \\\\\n\\midrule\n"

    for model_name, row in summary.iterrows():
        latex_str += f"{model_name}"
        for dist_name in dist_order:
            ll_mean, ll_std = row[('ll_mean', dist_name)], row[('ll_std', dist_name)]
            kl_mean, kl_std = row[('kl_mean', dist_name)], row[('kl_std', dist_name)]
            
            ll_str = f"${ll_mean:.3f} \\pm {ll_std:.3f}$"
            kl_str = f"${kl_mean:.3f} \\pm {kl_std:.3f}$"
            
            if ll_mean == summary['ll_mean'][dist_name].max(): ll_str = f"\\textbf{{{ll_str}}}"
            if kl_mean == summary['kl_mean'][dist_name].min(): kl_str = f"\\textbf{{{kl_str}}}"
            
            latex_str += f" & {ll_str} & {kl_str}"
        latex_str += " \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}"

    table_path = CONFIG['output_dir_results'] / "synthetic_dynamics.tex"
    with open(table_path, 'w') as f: f.write(latex_str)
    print(f"\nLaTeX Table saved to '{table_path}'")

def create_dynamics_figure(results_df: pd.DataFrame):
    """Generates the two-panel learning dynamics figure."""
    print("\n--- Generating Figure: Learning Dynamics on Noisy Data ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    summary = results_df.groupby(['Distribution', 'Model', 'Epoch']).agg(
        ll_mean=('Test Log-Likelihood', 'mean'), ll_std=('Test Log-Likelihood', 'std'),
        kl_mean=('Tail KL Divergence', 'mean'), kl_std=('Tail KL Divergence', 'std')
    ).reset_index()

    model_order = ['NSB (Ours)', 'Softmax NN (Fair)', 'Softmax NN', 'Negative Binomial', 'Poisson']
    colors = {'NSB (Ours)': NSB_COLORS['nsb'], 'Softmax NN (Fair)': NSB_COLORS['softmax_nn_(fair)'], 'Softmax NN': NSB_COLORS['softmax_nn'],
              'Negative Binomial': NSB_COLORS['negative_binomial'], 'Poisson': NSB_COLORS['poisson']}
    linestyles = {'Test Log-Likelihood': '-', 'Tail KL Divergence': '--'}

    for i, dist_name in enumerate(CONFIG['distributions'].keys()):
        ax = axes[i]
        ax2 = ax.twinx() # Create a second y-axis

        for model_name in model_order:
            model_data = summary[(summary['Distribution'] == dist_name) & (summary['Model'] == model_name)]
            
            # Plot Test LL on the left axis
            ax.plot(model_data['Epoch'], model_data['ll_mean'], color=colors[model_name], linestyle=linestyles['Test Log-Likelihood'], label=model_name)
            ax.fill_between(model_data['Epoch'], model_data['ll_mean'] - model_data['ll_std'], model_data['ll_mean'] + model_data['ll_std'], color=colors[model_name], alpha=0.1)
            
            # Plot Tail KL on the right axis
            ax2.plot(model_data['Epoch'], model_data['kl_mean'], color=colors[model_name], linestyle=linestyles['Tail KL Divergence'])
            ax2.fill_between(model_data['Epoch'], model_data['kl_mean'] - model_data['kl_std'], model_data['kl_mean'] + model_data['kl_std'], color=colors[model_name], alpha=0.1)

        ax.set_title(f"({chr(97 + i)}) {dist_name}", weight='bold')
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Test Log-Likelihood (Higher is better)", weight='bold', color='black')
        ax2.set_ylabel("Tail KL Divergence (Lower is better)", weight='bold', color='gray')
        ax2.set_yscale('log')
        
        # Create a unified legend
        lines, labels = ax.get_legend_handles_labels()
        fig.legend(lines, labels, loc='lower center', ncol=len(model_order), bbox_to_anchor=(0.5, -0.08))

    save_figure(fig, "synthetic_dynamics")

if __name__ == "__main__":
    results_df = run_experiments()
    create_results_table(results_df)
    create_dynamics_figure(results_df)


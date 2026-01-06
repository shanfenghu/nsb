"""
This script runs experiments to track the performance of different models,
epoch by epoch, on challenging, heavy-tailed synthetic distributions.

The script generates:
1.  A CSV file with the full learning curve results (`results/synthetic_dynamics.csv`).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import poisson, nbinom
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from nsb.gru_model import NSBGRU
from nsb.lstm_model import NSBLSTM
from nsb.attention_model import NSBAttention
from nsb.poisson import PoissonMLE
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.softmax_nn import SoftmaxNN

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
            # Hidden dimensions from count_parameters.py for fair parameter-matched comparison
            # NSB (Ours): hidden_dim=64 (4,289 params)
            # NSBGRU: hidden_dim=27 (4,591 params)
            # NSBLSTM: hidden_dim=23 (4,486 params)
            # NSBAttention: hidden_dim=19, max_k=150 (4,409 params)
            # SoftmaxNN (Fair): hidden_dim=27 (original, kept for reproducibility)
            fair_hidden_dim = 27  # Original value, kept for reproducibility
            nn_models = {
                'NSB': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
                'NSB-GRU': NSBGRU(hidden_dim=27),
                'NSB-LSTM': NSBLSTM(hidden_dim=23),
                'NSB-Attention': NSBAttention(hidden_dim=19, num_heads=1, max_k=CONFIG['nn_params']['k_max']),
                'Softmax NN (Fair)': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'], hidden_dim=fair_hidden_dim),
                'Softmax NN': SoftmaxNN(k_max=CONFIG['nn_params']['k_max'])
            }
            
            for model_name, model in nn_models.items():
                # Set up optimizer based on model type
                # Check NSBAttention first since it inherits from NSB
                if isinstance(model, NSBAttention):
                    optimizer = torch.optim.Adam(
                        list(model.attention.parameters()) + 
                        list(model.fc_pi.parameters()) + 
                        [model.pos_encoder], 
                        lr=CONFIG['nn_params']['lr']
                    )
                elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
                    optimizer = torch.optim.Adam(list(model.cell.parameters()) + [model.h0], lr=CONFIG['nn_params']['lr'])
                else:  # SoftmaxNN
                    optimizer = torch.optim.Adam(model.model.parameters(), lr=CONFIG['nn_params']['lr'])
                
                loader = DataLoader(TensorDataset(torch.from_numpy(train_data).long()), batch_size=CONFIG['nn_params']['batch_size'], shuffle=True)
                
                for epoch in range(CONFIG['nn_params']['epochs'] + 1):
                    if epoch > 0: # Training step
                        # Check NSBAttention first since it inherits from NSB
                        if isinstance(model, NSBAttention):
                            model.attention.train()
                            model.fc_pi.train()
                        elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
                            model.cell.train()
                        else:  # SoftmaxNN
                            model.model.train()
                        
                        for batch_counts in loader:
                            optimizer.zero_grad()
                            # Check NSBAttention first since it inherits from NSB
                            if isinstance(model, NSBAttention):
                                log_probs = model._compute_log_probs(batch_counts[0].to(model.device))
                                loss = -log_probs.mean()
                            elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
                                log_probs = model._compute_log_probs(batch_counts[0].to(model.device))
                                loss = -log_probs.mean()
                            else:  # SoftmaxNN
                                dummy_input = torch.zeros(len(batch_counts[0]), 1)
                                logits = model.model(dummy_input)
                                loss = torch.nn.functional.cross_entropy(logits, batch_counts[0])
                            loss.backward()
                            optimizer.step()
                    
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
                    elif isinstance(model, NSBAttention):
                        # NSBAttention is limited by max_k, so cap k_max_eval to model.max_k
                        k_max_for_pmf = min(CONFIG['k_max_eval'], model.max_k)
                        learned_pmf_partial = model.predict_pmf(k_max=k_max_for_pmf)
                        # Pad with zeros to match k_max_eval size
                        learned_pmf = np.zeros(CONFIG['k_max_eval'] + 1)
                        len_to_copy = min(len(learned_pmf_partial), len(learned_pmf))
                        learned_pmf[:len_to_copy] = learned_pmf_partial[:len_to_copy]
                    else:
                        learned_pmf = model.predict_pmf(k_max=CONFIG['k_max_eval'])
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

if __name__ == "__main__":
    run_experiments()


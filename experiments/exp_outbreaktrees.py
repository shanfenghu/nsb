"""
This script runs experiments to train and evaluate the NSB model and all baselines
on the processed `outbreaktrees` dataset of SARS and MERS transmission events.

The script generates:
1.  A CSV file with the raw results (`results/real_world_results.csv`).
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import time

# --- Local Imports ---
from nsb.model import NSB
from nsb.gru_model import NSBGRU
from nsb.lstm_model import NSBLSTM
from nsb.attention_model import NSBAttention
from nsb.poisson import PoissonMLE
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.softmax_nn import SoftmaxNN
from count_parameters import count_parameters

# --- Configuration ---
CONFIG = {
    'seeds': list(range(20)),  # Increased to 20 for better statistical power
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
        
        # Hidden dimensions from count_parameters.py for fair parameter-matched comparison
        # NSB: hidden_dim=64 (4,289 params)
        # NSBGRU: hidden_dim=27 (4,591 params)
        # NSBLSTM: hidden_dim=23 (4,486 params)
        # NSBAttention: hidden_dim=19, max_k=150 (4,409 params)
        # SoftmaxNN (Fair): hidden_dim=28 (4,435 params)
        fair_hidden_dim = 28
        models = {
            'NSB': NSB(hidden_dim=CONFIG['nn_params']['hidden_dim']),
            'NSB-GRU': NSBGRU(hidden_dim=27),
            'NSB-LSTM': NSBLSTM(hidden_dim=23),
            'NSB-Attention': NSBAttention(hidden_dim=19, num_heads=1, max_k=CONFIG['nn_params']['k_max']),
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
                    # Check NSBAttention first since it inherits from NSB
                    if isinstance(model, NSBAttention):
                        model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
                    elif isinstance(model, (NSB, NSBGRU, NSBLSTM, SoftmaxNN)):
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
                    # Check NSBAttention first since it inherits from NSB
                    if isinstance(model, NSBAttention):
                        # NSBAttention is limited by max_k, so cap k_max_eval to model.max_k
                        k_max_for_pmf = min(k_max_eval, model.max_k)
                        learned_pmf_partial = model.predict_pmf(k_max=k_max_for_pmf)
                        # Pad with zeros to match k_max_eval size
                        learned_pmf = np.zeros(k_max_eval + 1)
                        len_to_copy = min(len(learned_pmf_partial), len(learned_pmf))
                        learned_pmf[:len_to_copy] = learned_pmf_partial[:len_to_copy]
                    elif isinstance(model, SoftmaxNN):
                        pmf_full = model.predict_pmf()
                        learned_pmf = np.zeros(k_max_eval + 1)
                        len_to_copy = min(len(pmf_full), len(learned_pmf))
                        learned_pmf[:len_to_copy] = pmf_full[:len_to_copy]
                    elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
                        learned_pmf = model.predict_pmf(k_max=k_max_eval)
                    else: # For Poisson and NegativeBinomial
                        learned_pmf = model.pmf(np.arange(k_max_eval + 1))
                    
                    learned_tail_pmf = learned_pmf[tail_start_idx:]
                    if learned_tail_pmf.sum() > 1e-9: # Avoid division by zero
                        learned_tail_pmf = learned_tail_pmf / learned_tail_pmf.sum()
                        tail_kl = kl_divergence(empirical_tail_pmf, learned_tail_pmf)

                # Count parameters using the comprehensive count_parameters function
                num_params = count_parameters(model)

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

if __name__ == "__main__":
    run_experiment()
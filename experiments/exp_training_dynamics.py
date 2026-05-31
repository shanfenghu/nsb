"""
exp_training_dynamics.py: Training Dynamics Analysis

Generates training and test loss curves for the NSB model on the SARS/MERS
outbreaktrees dataset. This script demonstrates:
1. Training convergence over epochs
2. Generalization gap (train vs test loss)
3. Model stability across multiple random seeds

The figure shows:
- Training and test negative log-likelihood over epochs
- Test log-likelihood convergence (zoomed view of final epochs)
- Training stability across 5 random seeds (mean ± std)

This analysis validates that the model converges stably and generalizes well
to unseen data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Internal module imports
from nsb.model import NSB
from plot_utils import setup_plot_style, save_figure, NSB_COLORS

# REPRODUCIBILITY SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration
CONFIG = {
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'test_size': 0.2,
    'output_dir_figures': Path("figures"),
    'output_dir_results': Path("results"),
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    },
    'n_seeds': 20,  # Number of random seeds for stability analysis
    'seeds': list(range(20))  # Seeds [0, 1, ..., 19] matching exp_outbreaktrees.py for consistency
}

# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------

def load_and_split_data(path: Path, test_size: float, seed: int):
    """
    Loads offspring count data from CSV and splits into train/test sets.
    
    Args:
        path: Path to CSV file containing 'offspring_count' column
        test_size: Fraction of data to use for testing (e.g., 0.2 for 80/20 split)
        seed: Random seed for reproducible train/test split
        
    Returns:
        tuple: (train_data, test_data) as numpy arrays
    """
    df = pd.read_csv(path)
    counts = df['offspring_count'].values
    return train_test_split(counts, test_size=test_size, random_state=seed)

# --------------------------------------------------------------------------
# 2. TRAINING WITH LOSS TRACKING
# --------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculates the KL divergence D_KL(P || Q) for discrete distributions.
    
    Args:
        p: True/reference distribution
        q: Approximating distribution
        
    Returns:
        float: KL divergence value
    """
    # Add small epsilon to avoid log(0) and division by zero
    p = p + 1e-10
    q = q + 1e-10
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def compute_empirical_pmf(data: np.ndarray, k_max: int) -> np.ndarray:
    """
    Computes the empirical PMF from data.
    
    Args:
        data: Array of observed counts
        k_max: Maximum count to include
        
    Returns:
        np.ndarray: Empirical PMF of length (k_max + 1)
    """
    pmf = np.zeros(k_max + 1)
    for count in data:
        if count <= k_max:
            pmf[int(count)] += 1
    pmf = pmf / len(data)  # Normalize
    return pmf

def train_with_tracking(model: NSB, train_data: np.ndarray, test_data: np.ndarray, 
                       config: dict) -> dict:
    """
    Trains the NSB model while tracking training and test loss over epochs.
    Also tracks tail KL divergence on test data during training.
    
    Args:
        model: NSB model instance to train
        train_data: Training data (offspring counts)
        test_data: Test data (offspring counts)
        config: Configuration dictionary with 'nn_params'
        
    Returns:
        dict: Contains 'train_losses', 'test_losses', 'test_log_likelihoods', 'tail_kl_divergences' arrays
    """
    from torch.utils.data import TensorDataset, DataLoader
    from torch import optim
    
    epochs = config['nn_params']['epochs']
    lr = config['nn_params']['lr']
    batch_size = config['nn_params']['batch_size']
    k_max_eval = 150  # For PMF evaluation
    
    # Compute empirical PMF from test data (ground truth for tail KL)
    empirical_pmf_test = compute_empirical_pmf(test_data, k_max_eval)
    median_val = int(np.median(test_data))
    tail_start_idx = median_val + 1
    empirical_tail_pmf_test = empirical_pmf_test[tail_start_idx:]
    empirical_tail_pmf_test = empirical_tail_pmf_test / empirical_tail_pmf_test.sum()  # Normalize tail
    
    # Setup data loaders
    train_counts = torch.from_numpy(train_data).long()
    train_dataset = TensorDataset(train_counts)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_counts = torch.from_numpy(test_data).long()
    test_dataset = TensorDataset(test_counts)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = optim.Adam(list(model.cell.parameters()) + [model.h0], lr=lr)
    
    # Track losses
    train_losses = []
    test_losses = []
    test_log_likelihoods = []
    tail_kl_divergences = []
    
    model.cell.train()
    
    for epoch in range(epochs):
        # Training phase
        epoch_train_loss = 0.0
        n_train_batches = 0
        
        for batch_counts in train_loader:
            batch_counts = batch_counts[0].to(model.device)
            optimizer.zero_grad()
            
            log_probs = model._compute_log_probs(batch_counts)
            loss = -log_probs.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.cell.eval()
        epoch_test_loss = 0.0
        epoch_test_log_likelihood = 0.0
        n_test_batches = 0
        
        with torch.no_grad():
            for batch_counts in test_loader:
                batch_counts = batch_counts[0].to(model.device)
                log_probs = model._compute_log_probs(batch_counts)
                loss = -log_probs.mean()
                log_likelihood = log_probs.mean()
                
                epoch_test_loss += loss.item()
                epoch_test_log_likelihood += log_likelihood.item()
                n_test_batches += 1
        
        avg_test_loss = epoch_test_loss / n_test_batches
        avg_test_log_likelihood = epoch_test_log_likelihood / n_test_batches
        
        test_losses.append(avg_test_loss)
        test_log_likelihoods.append(avg_test_log_likelihood)
        
        # Compute tail KL divergence on test data
        learned_pmf = model.predict_pmf(k_max=k_max_eval)
        learned_tail_pmf = learned_pmf[tail_start_idx:]
        learned_tail_pmf = learned_tail_pmf / learned_tail_pmf.sum()  # Normalize tail
        tail_kl = kl_divergence(empirical_tail_pmf_test, learned_tail_pmf)
        tail_kl_divergences.append(tail_kl)
        
        model.cell.train()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, "
                  f"Test Loss = {avg_test_loss:.4f}, Test LL = {avg_test_log_likelihood:.4f}, "
                  f"Tail KL = {tail_kl:.4f}")
    
    return {
        'train_losses': np.array(train_losses),
        'test_losses': np.array(test_losses),
        'test_log_likelihoods': np.array(test_log_likelihoods),
        'tail_kl_divergences': np.array(tail_kl_divergences)
    }

# --------------------------------------------------------------------------
# 3. MULTI-SEED STABILITY ANALYSIS
# --------------------------------------------------------------------------

def run_stability_analysis(train_data: np.ndarray, test_data: np.ndarray, 
                          config: dict) -> dict:
    """
    Runs training across multiple random seeds to assess stability.
    Uses the same train/test split for all seeds (only model initialization varies).
    
    Args:
        train_data: Training data
        test_data: Test data
        config: Configuration dictionary
        
    Returns:
        dict: Contains arrays of losses across seeds (mean, std, individual runs)
    """
    all_train_losses = []
    all_test_losses = []
    all_test_log_likelihoods = []
    all_tail_kl_divergences = []
    
    for seed in tqdm(config['seeds'], desc="Training across seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = NSB(hidden_dim=config['nn_params']['hidden_dim'])
        history = train_with_tracking(model, train_data, test_data, config)
        
        all_train_losses.append(history['train_losses'])
        all_test_losses.append(history['test_losses'])
        all_test_log_likelihoods.append(history['test_log_likelihoods'])
        all_tail_kl_divergences.append(history['tail_kl_divergences'])
    
    # Convert to numpy arrays
    all_train_losses = np.array(all_train_losses)  # Shape: (n_seeds, n_epochs)
    all_test_losses = np.array(all_test_losses)
    all_test_log_likelihoods = np.array(all_test_log_likelihoods)
    all_tail_kl_divergences = np.array(all_tail_kl_divergences)
    
    return {
        'train_losses_mean': np.mean(all_train_losses, axis=0),
        'train_losses_std': np.std(all_train_losses, axis=0),
        'test_losses_mean': np.mean(all_test_losses, axis=0),
        'test_losses_std': np.std(all_test_losses, axis=0),
        'test_log_likelihoods_mean': np.mean(all_test_log_likelihoods, axis=0),
        'test_log_likelihoods_std': np.std(all_test_log_likelihoods, axis=0),
        'tail_kl_divergences_mean': np.mean(all_tail_kl_divergences, axis=0),
        'tail_kl_divergences_std': np.std(all_tail_kl_divergences, axis=0),
        'all_train_losses': all_train_losses,
        'all_test_losses': all_test_losses,
        'all_test_log_likelihoods': all_test_log_likelihoods,
        'all_tail_kl_divergences': all_tail_kl_divergences
    }

def run_stability_analysis_with_splits(config: dict) -> dict:
    """
    Runs training across multiple random seeds with different train/test splits per seed.
    This matches the methodology used in exp_outbreaktrees.py for consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Contains arrays of losses across seeds (mean, std, individual runs)
    """
    all_train_losses = []
    all_test_losses = []
    all_test_log_likelihoods = []
    all_tail_kl_divergences = []
    
    for seed in tqdm(config['seeds'], desc="Training across seeds"):
        # Each seed gets its own train/test split (matching exp_outbreaktrees.py)
        train_data, test_data = load_and_split_data(
            config['data_path'], config['test_size'], seed
        )
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = NSB(hidden_dim=config['nn_params']['hidden_dim'])
        history = train_with_tracking(model, train_data, test_data, config)
        
        all_train_losses.append(history['train_losses'])
        all_test_losses.append(history['test_losses'])
        all_test_log_likelihoods.append(history['test_log_likelihoods'])
        all_tail_kl_divergences.append(history['tail_kl_divergences'])
    
    # Convert to numpy arrays
    all_train_losses = np.array(all_train_losses)  # Shape: (n_seeds, n_epochs)
    all_test_losses = np.array(all_test_losses)
    all_test_log_likelihoods = np.array(all_test_log_likelihoods)
    all_tail_kl_divergences = np.array(all_tail_kl_divergences)
    
    return {
        'train_losses_mean': np.mean(all_train_losses, axis=0),
        'train_losses_std': np.std(all_train_losses, axis=0),
        'test_losses_mean': np.mean(all_test_losses, axis=0),
        'test_losses_std': np.std(all_test_losses, axis=0),
        'test_log_likelihoods_mean': np.mean(all_test_log_likelihoods, axis=0),
        'test_log_likelihoods_std': np.std(all_test_log_likelihoods, axis=0),
        'tail_kl_divergences_mean': np.mean(all_tail_kl_divergences, axis=0),
        'tail_kl_divergences_std': np.std(all_tail_kl_divergences, axis=0),
        'all_train_losses': all_train_losses,
        'all_test_losses': all_test_losses,
        'all_test_log_likelihoods': all_test_log_likelihoods,
        'all_tail_kl_divergences': all_tail_kl_divergences
    }

# --------------------------------------------------------------------------
# 4. PLOTTING
# --------------------------------------------------------------------------

def plot_training_dynamics(history: dict, stability: dict, config: dict):
    """
    Creates a 3-panel figure showing training dynamics.
    
    Args:
        history: Training history from single run (seed=42)
        stability: Stability analysis results across multiple seeds
        config: Configuration dictionary
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], hspace=0.25, wspace=0.2)
    
    epochs = np.arange(1, config['nn_params']['epochs'] + 1)
    
    # Training and test loss curves
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(epochs, history['train_losses'], 'o-', color=NSB_COLORS['nsb'], 
              linewidth=2.5, markersize=4, label='Training Loss', alpha=0.8)
    ax_a.plot(epochs, history['test_losses'], 's-', color='#DC143C', 
              linewidth=2.5, markersize=4, label='Test Loss', alpha=0.8)
    ax_a.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax_a.set_ylabel('Negative Log-Likelihood', fontweight='bold', fontsize=12)
    ax_a.set_title('(a) Training and Test Loss', fontweight='bold', fontsize=13)
    ax_a.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    ax_a.set_xlim(0, config['nn_params']['epochs'] + 1)
    
    # Stability across random seeds
    ax_b = fig.add_subplot(gs[0, 1])
    mean_ll = stability['test_log_likelihoods_mean']
    std_ll = stability['test_log_likelihoods_std']
    
    # Plot mean with shaded std region
    ax_b.plot(epochs, mean_ll, '-', color=NSB_COLORS['nsb'], 
              linewidth=3, label='Mean (20 seeds)', zorder=3)
    ax_b.fill_between(epochs, mean_ll - std_ll, mean_ll + std_ll, 
                      color=NSB_COLORS['nsb'], alpha=0.2, 
                      label='±1 Std Dev', zorder=2)
    
    # Plot individual runs (lighter)
    for i, seed in enumerate(config['seeds']):
        ax_b.plot(epochs, stability['all_test_log_likelihoods'][i], 
                 '-', color=NSB_COLORS['nsb'], alpha=0.15, linewidth=1, zorder=1)
    
    ax_b.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax_b.set_ylabel('Test Log-Likelihood', fontweight='bold', fontsize=12)
    ax_b.set_title('(b) Training Stability across Seeds', fontweight='bold', fontsize=13)
    ax_b.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax_b.grid(True, alpha=0.3, linestyle='--')
    ax_b.set_xlim(0, config['nn_params']['epochs'] + 1)
    
    # Tail KL divergence
    ax_c = fig.add_subplot(gs[0, 2])
    mean_tail_kl = stability['tail_kl_divergences_mean']
    std_tail_kl = stability['tail_kl_divergences_std']
    
    # Plot mean with shaded std region (using same color as sample complexity experiment)
    ax_c.plot(epochs, mean_tail_kl, '-', color=NSB_COLORS['nsb_subcritical'], 
              linewidth=3, label='Mean (20 seeds)', zorder=3)
    ax_c.fill_between(epochs, mean_tail_kl - std_tail_kl, mean_tail_kl + std_tail_kl, 
                      color=NSB_COLORS['nsb_subcritical'], alpha=0.2, 
                      label='±1 Std Dev', zorder=2)
    
    # Plot individual runs (lighter)
    for i, seed in enumerate(config['seeds']):
        ax_c.plot(epochs, stability['all_tail_kl_divergences'][i], 
                 '-', color=NSB_COLORS['nsb_subcritical'], alpha=0.15, linewidth=1, zorder=1)
    
    ax_c.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax_c.set_ylabel('Tail KL Divergence', fontweight='bold', fontsize=12)
    ax_c.set_title('(c) Tail Learning Performance', fontweight='bold', fontsize=13)
    ax_c.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax_c.grid(True, alpha=0.3, linestyle='--')
    ax_c.set_xlim(0, config['nn_params']['epochs'] + 1)
    
    return fig

# --------------------------------------------------------------------------
# 5. MAIN EXPERIMENT
# --------------------------------------------------------------------------

def main():
    """Main experiment function."""
    print("=" * 70)
    print("Training Dynamics Analysis")
    print("=" * 70)
    
    # Load data (use seed=42 for single run visualization)
    print(f"\n1. Loading data from {CONFIG['data_path']}...")
    train_data_single, test_data_single = load_and_split_data(
        CONFIG['data_path'], CONFIG['test_size'], SEED
    )
    print(f"   Training samples: {len(train_data_single)}")
    print(f"   Test samples: {len(test_data_single)}")
    
    # Single run for detailed curves (using seed=42 split)
    print(f"\n2. Training NSB model (single run, seed={SEED})...")
    model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
    history = train_with_tracking(model, train_data_single, test_data_single, CONFIG)
    
    # Multi-seed stability analysis (each seed gets its own train/test split)
    print(f"\n3. Running stability analysis across {CONFIG['n_seeds']} seeds...")
    print("   Note: Each seed uses a different train/test split for robustness.")
    stability = run_stability_analysis_with_splits(CONFIG)
    
    # Save results
    print(f"\n4. Saving results...")
    CONFIG['output_dir_results'].mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'epoch': np.arange(1, CONFIG['nn_params']['epochs'] + 1),
        'train_loss': history['train_losses'],
        'test_loss': history['test_losses'],
        'test_log_likelihood': history['test_log_likelihoods'],
        'tail_kl_divergence': history['tail_kl_divergences'],
        'test_log_likelihood_mean': stability['test_log_likelihoods_mean'],
        'test_log_likelihood_std': stability['test_log_likelihoods_std'],
        'tail_kl_divergence_mean': stability['tail_kl_divergences_mean'],
        'tail_kl_divergence_std': stability['tail_kl_divergences_std']
    })
    results_df.to_csv(CONFIG['output_dir_results'] / 'training_dynamics.csv', index=False)
    print(f"   Results saved to {CONFIG['output_dir_results'] / 'training_dynamics.csv'}")
    
    # Generate figure
    print(f"\n5. Generating training dynamics figure...")
    fig = plot_training_dynamics(history, stability, CONFIG)
    save_figure(fig, 'training_dynamics', str(CONFIG['output_dir_figures']))
    
    print("\n" + "=" * 70)
    print("Training dynamics analysis complete!")
    print("=" * 70)
    print(f"\nFinal metrics (seed={SEED}):")
    print(f"  Final Training Loss: {history['train_losses'][-1]:.4f}")
    print(f"  Final Test Loss: {history['test_losses'][-1]:.4f}")
    print(f"  Final Test Log-Likelihood: {history['test_log_likelihoods'][-1]:.4f}")
    print(f"\nStability metrics (mean ± std across {CONFIG['n_seeds']} seeds):")
    print(f"  Final Test Log-Likelihood: {stability['test_log_likelihoods_mean'][-1]:.4f} ± {stability['test_log_likelihoods_std'][-1]:.4f}")

if __name__ == "__main__":
    main()

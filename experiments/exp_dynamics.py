"""
Experiment: Visualizing the Learned Dynamics of the NSB Model.

This script provides a deep, mechanistic insight into how the NSB model
learns to handle long-tailed distributions like those found in superspreading
events. It trains an NSB model on the real-world outbreak data and then
visualizes the trajectory of its internal hidden states.

The script generates a figure that illustrates:
1.  The structured "program" or manifold the RNN learns in its state space.
2.  The "stick conservation" strategy it employs along this path to allow for
    the possibility of rare, high-magnitude events.
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path

# --- Local Imports ---
from nsb.model import NSB
from plot_utils import setup_plot_style, save_figure

# --- Configuration ---
CONFIG = {
    'seed': 42, # Use one representative seed for this visualization
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'output_dir_figures': Path("figures"),
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    },
    'visualization_params': {
        'unroll_steps': 200, # How far to unroll the RNN to see the dynamics
        'tsne_perplexity': 30,
    }
}

def generate_trajectory(model: NSB, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Unrolls a trained NSB model to generate its hidden state and pi trajectory.

    Args:
        model (NSB): The trained NSB model.
        n_steps (int): The number of recursive steps to unroll.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_trajectory: The sequence of hidden states.
            - pi_trajectory: The sequence of break proportions.
    """
    model.cell.eval()
    with torch.no_grad():
        h = model.h0.cpu()
        h_trajectory = np.zeros((n_steps, model.hidden_dim))
        pi_trajectory = np.zeros(n_steps)

        for k in range(n_steps):
            h_trajectory[k, :] = h.numpy()
            h, pi_logit = model.cell(h)
            pi_k = torch.sigmoid(pi_logit).item()
            pi_trajectory[k] = pi_k
            
    return h_trajectory, pi_trajectory

def create_dynamics_figure(h_trajectory: np.ndarray, pi_trajectory: np.ndarray):
    """
    Generates and saves the figure visualizing the learned dynamics.
    """
    print("\n--- Generating Figure: Visualizing Learned Dynamics ---")
    setup_plot_style()

    # --- Step 1: Dimensionality Reduction ---
    print("Running t-SNE on hidden state trajectory...")
    tsne = TSNE(
        n_components=2,
        perplexity=CONFIG['visualization_params']['tsne_perplexity'],
        random_state=CONFIG['seed'],
        init='pca',
        learning_rate='auto'
    )
    h_2d = tsne.fit_transform(h_trajectory)

    # --- Step 2: Create the Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    # fig.suptitle("The Learned Dynamics of Stick Conservation for Superspreading", fontsize=18, weight='bold')

    # --- Learned State-Space Manifold ---
    ax1 = axes[0]
    # --- MODIFICATION: Add faint connecting lines ---
    ax1.plot(h_2d[:, 0], h_2d[:, 1], '-', color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    scatter1 = ax1.scatter(
        h_2d[:, 0], h_2d[:, 1],
        c=np.arange(len(h_2d)), # Color by step number k
        cmap='viridis',
        alpha=0.8,
        s=50,
        zorder=2 # Ensure points are drawn on top of the line
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Recursive Step (k)", weight='bold')
    ax1.set_title("(a) The Learned State-Space Manifold", weight='bold')
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- Stick-Conservation Mechanism ---
    ax2 = axes[1]
    # --- MODIFICATION: Add faint connecting lines ---
    ax2.plot(h_2d[:, 0], h_2d[:, 1], '-', color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    scatter2 = ax2.scatter(
        h_2d[:, 0], h_2d[:, 1],
        c=pi_trajectory, # Color by break proportion pi_k
        cmap='magma_r', # Use a reverse map so small values are dark
        alpha=0.8,
        s=50,
        zorder=2 # Ensure points are drawn on top of the line
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Break Proportion ($\pi_k$)", weight='bold')
    ax2.set_title("(b) The Stick-Conservation Mechanism", weight='bold')
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.set_xticks([])
    ax2.set_yticks([])

    save_figure(fig, "learned_dynamics")

def main():
    """Main function to train the model and generate the visualization."""
    # --- Load Data ---
    print(f"Loading real-world data from '{CONFIG['data_path']}'...")
    df = pd.read_csv(CONFIG['data_path'])
    train_data = df['offspring_count'].values

    # --- Train a Representative NSB Model ---
    print("Training a representative NSB model on the full dataset...")
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    model = NSB(hidden_dim=CONFIG['nn_params']['hidden_dim'])
    # Move model to CPU for trajectory generation to ensure consistency
    model.cell.to('cpu')
    model.h0 = model.h0.to('cpu')
    
    model.fit(
        train_data,
        epochs=CONFIG['nn_params']['epochs'],
        lr=CONFIG['nn_params']['lr'],
        batch_size=CONFIG['nn_params']['batch_size']
    )

    # --- Generate and Visualize the Dynamics ---
    h_trajectory, pi_trajectory = generate_trajectory(
        model,
        n_steps=CONFIG['visualization_params']['unroll_steps']
    )
    
    create_dynamics_figure(h_trajectory, pi_trajectory)

if __name__ == "__main__":
    main()

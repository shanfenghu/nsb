"""
This script generates plots and tables from the real-world experiment results.

The script generates:
1.  A LaTeX table for the main benchmark results (Table 2).
2.  A figure with multiple panels analyzing the results:
    - Panels (a), (b), (c): Distribution analysis including a rank-frequency
      plot with a spectral analysis inset
    - Panels (d), (e): Dynamics visualization showing the learned state-space
      manifold and stick-conservation mechanism
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
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
    },
    'visualization_params': {
        'unroll_steps': 200, # How far to unroll the RNN to see the dynamics
        'tsne_perplexity': 30,
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

def fit_with_eigenvalue_tracking(model: NSB, data: np.ndarray, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> list[complex]:
    """
    Trains the NSB model while tracking the dominant eigenvalue at each epoch.
    
    Args:
        model (NSB): The NSB model to train.
        data (np.ndarray): Training data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
    
    Returns:
        list[complex]: List of dominant eigenvalues at each epoch (including epoch 0).
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    from tqdm import tqdm
    
    counts_tensor = torch.from_numpy(data).long()
    dataset = TensorDataset(counts_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(list(model.cell.parameters()) + [model.h0], lr=lr)
    
    # Track eigenvalues: start with initial state (epoch 0)
    eigenvalue_history = []
    Wh_initial = model.cell.fc_h.weight.data.detach().cpu().numpy()
    eigenvalues_initial = np.linalg.eigvals(Wh_initial)
    idx_dom = np.argmax(np.abs(eigenvalues_initial))
    eigenvalue_history.append(eigenvalues_initial[idx_dom])
    
    model.cell.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_counts in pbar:
            batch_counts = batch_counts[0].to(model.device)
            optimizer.zero_grad()
            
            log_probs = model._compute_log_probs(batch_counts)
            loss = -log_probs.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Track dominant eigenvalue after each epoch
        Wh = model.cell.fc_h.weight.data.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(Wh)
        idx_dom = np.argmax(np.abs(eigenvalues))
        eigenvalue_history.append(eigenvalues[idx_dom])
    
    return eigenvalue_history

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
    """Generates the figure for the real-world analysis with dynamics panel."""
    print("\n--- Generating Figure: Real-World Distribution Analysis ---")
    setup_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.5), constrained_layout=True)
    
    plot_seed = CONFIG['seeds'][0]
    train_data, test_data = load_and_split_data(CONFIG['data_path'], CONFIG['test_size'], plot_seed)
    
    # Set seed BEFORE creating models to ensure reproducible initialization
    torch.manual_seed(plot_seed)
    np.random.seed(plot_seed)
    
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
    # Track eigenvalue evolution for NSB model only
    nsb_eigenvalue_history = None
    
    for model_name, model in models_to_plot.items():
        # Reset seed for each model to ensure reproducible training
        torch.manual_seed(plot_seed)
        np.random.seed(plot_seed)
        # Check NSBAttention first since it inherits from NSB
        if isinstance(model, NSBAttention):
            model.fit(train_data, epochs=CONFIG['nn_params']['epochs'], lr=CONFIG['nn_params']['lr'])
        elif isinstance(model, NSB) and model_name == 'NSB':
            # Track eigenvalues during NSB training
            nsb_eigenvalue_history = fit_with_eigenvalue_tracking(
                model, train_data, 
                epochs=CONFIG['nn_params']['epochs'], 
                lr=CONFIG['nn_params']['lr']
            )
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
    
    # Draw vertical red line at x=1.0 to indicate critical boundary
    axins.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
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
    
    # Set axis labels (same format as panel d)
    axins.set_xlabel('Re(λ)', fontsize=9, labelpad=2)
    axins.set_ylabel('Im(λ)', fontsize=9, labelpad=2)
    
    # Set ticks to show a few key values
    axins.set_xticks([-1, 0, 1])
    axins.set_yticks([-1, 0, 1])
    axins.tick_params(labelsize=7)
    
    axins.set_title("NSB's Spectral Radius", fontsize=10)
    axins.set_aspect('equal', adjustable='box')

    # --- Panel (d): Dynamics Visualization ---
    ax4 = axes[3]
    # Use the NSB model already trained for panels a-c (from models_to_plot)
    # This ensures consistency across all panels
    nsb_model = models_to_plot['NSB']
    print("Generating NSB trajectory for dynamics visualization using model from panels a-c...")
    
    # Move model to CPU for trajectory generation to ensure consistency
    nsb_model.cell.to('cpu')
    nsb_model.h0 = nsb_model.h0.to('cpu')
    
    h_trajectory, pi_trajectory = generate_trajectory(
        nsb_model,
        n_steps=CONFIG['visualization_params']['unroll_steps']
    )
    print(f"Trajectory shape: {h_trajectory.shape}, pi_trajectory shape: {pi_trajectory.shape}")
    print(f"Hidden state stats: min={h_trajectory.min():.4f}, max={h_trajectory.max():.4f}, mean={h_trajectory.mean():.4f}, std={h_trajectory.std():.4f}")
    
    # Calculate L2 norms of hidden states
    h_norm = np.linalg.norm(h_trajectory, axis=1)  # L2 norm of each hidden state
    
    # Calculate ratio: ||h_k|| / ||h_{k-1}|| for k >= 1
    # This shows how the hidden state norm changes between consecutive steps
    h_norm_ratio = np.zeros(len(h_trajectory))
    h_norm_ratio[0] = np.nan  # First step has no previous step
    h_norm_ratio[1:] = h_norm[1:] / h_norm[:-1]
    
    print(f"Hidden state norm ratio range: [{np.nanmin(h_norm_ratio):.4f}, {np.nanmax(h_norm_ratio):.4f}]")
    print(f"π_k range: [{pi_trajectory.min():.4f}, {pi_trajectory.max():.4f}]")
    
    # Find the point where norm ratio levels off to 1.0 (within tolerance)
    # Look for a longer stable period to find where it truly stabilizes after any initial dip
    tolerance = 0.01  # Tighter tolerance: within 3% of 1.0
    stable_window = 15  # Require stability for at least 15 consecutive steps
    k_stable = None
    for k in range(stable_window, len(h_norm_ratio)):
        window = h_norm_ratio[k-stable_window:k]
        # Check if all values in window are close to 1.0 and stable (low variance)
        if np.all(np.abs(window - 1.0) < tolerance):
            # Also check that the window has low variance (truly stable, not oscillating)
            window_std = np.nanstd(window)
            if window_std < 0.02:  # Low variance indicates stability
                k_stable = k - stable_window  # First k in the stable window
                break
    
    # If no stable window found, try with a slightly shorter window
    if k_stable is None:
        stable_window = 10
        for k in range(stable_window, len(h_norm_ratio)):
            window = h_norm_ratio[k-stable_window:k]
            if np.all(np.abs(window - 1.0) < tolerance):
                window_std = np.nanstd(window)
                if window_std < 0.02:
                    k_stable = k - stable_window
                    break
    
    # If still no stable window found, use the point closest to 1.0 after k=7
    if k_stable is None:
        # Look for the point after k=7 that's closest to 1.0
        mask = np.arange(len(h_norm_ratio)) >= 7
        if np.any(mask):
            valid_indices = np.where(mask)[0]
            k_stable = valid_indices[np.nanargmin(np.abs(h_norm_ratio[mask] - 1.0))]
        else:
            k_stable = np.nanargmin(np.abs(h_norm_ratio - 1.0))
    
    print(f"Norm ratio stabilizes at k={k_stable}, ratio={h_norm_ratio[k_stable]:.4f}")
    
    # Panel (d): Dual-axis time series visualization
    # Left Y-axis: break proportion (π_k) - shows stick-conservation mechanism
    # Right Y-axis: hidden state norm ratio (||h_k|| / ||h_{k-1}||) - shows hidden state evolution
    # X-axis: unroll step (k)
    
    k_steps = np.arange(len(pi_trajectory))
    
    # Plot π_k on left axis (royal blue)
    royal_blue = '#4169E1'
    ax4.plot(k_steps, pi_trajectory, '-', color=royal_blue, linewidth=2, label=r'Break Proportion ($\pi_k$)', zorder=2)
    ax4.set_xlabel("Offspring Count/Unroll Step (k)", weight='bold')
    ax4.set_ylabel(r"Break Proportion ($\pi_k$)", weight='bold', color=royal_blue)
    ax4.tick_params(axis='y', labelcolor=royal_blue)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Create right axis for hidden state norm ratio (orangered)
    orangered = '#FF4500'
    ax4_right = ax4.twinx()
    ax4_right.plot(k_steps, h_norm_ratio, '-', color=orangered, linewidth=2, label=r'$||h_k|| / ||h_{k-1}||$', zorder=1, alpha=0.8)
    ax4_right.set_ylabel(r"Hidden State Norm Ratio ($||h_k|| / ||h_{k-1}||$)", weight='bold', color=orangered)
    ax4_right.tick_params(axis='y', labelcolor=orangered)
    
    # Add vertical line at the point where norm ratio levels off to 1.0
    if k_stable is not None and k_stable < len(k_steps):
        ax4.axvline(x=k_stable, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        # Add text annotation for the k value
        # Position annotation to the left and down to use empty space
        y_max = ax4.get_ylim()[1]
        y_min = ax4.get_ylim()[0]
        # Position text to the left of the line and in the lower portion
        text_x = k_stable + (ax4.get_xlim()[1] - ax4.get_xlim()[0]) * 0.05  # Left of the line
        text_y = y_min + (y_max - y_min) * 0.7  # Lower portion
        # Arrow points from text to the vertical line at a point on the line
        arrow_y = y_min + (y_max - y_min) * 0.6  # Mid-lower point on the line
        # Get the stabilized values at k_stable
        norm_ratio_stable = h_norm_ratio[k_stable]
        pi_stable = pi_trajectory[k_stable]
        ax4.annotate(
            f'Equilibrium reached at \n Offspring Count k={k_stable}\n'
            f'Norm Ratio: {norm_ratio_stable:.4f}\n'
            f'Break Proportion: {pi_stable:.4f}',
            xy=(k_stable, arrow_y),  # Arrow points to this point on the vertical line
            xytext=(text_x, text_y),  # Text position (left and down)
            fontsize=12, color='black', weight='bold',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8)
        )
    
    # Add legend (same fontsize as panels a-c, which use default)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_right.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax4.set_title("(d) NSB's Learned Dynamics/Equilibrium", weight='bold')
    
    # --- Inset: Dominant Eigenvalue Evolution During Training ---
    # Show how the dominant eigenvalue evolves from initialization to final trained state
    if nsb_eigenvalue_history is not None:
        # Position inset at middle bottom: [x0, y0, width, height] in axes coordinates
        axins_eig = ax4.inset_axes([0.3, 0.08, 0.45, 0.45])
        
        # Draw unit circle as reference (same linewidth as panel c)
        unit_circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1, zorder=1)
        axins_eig.add_patch(unit_circle)
        
        # Draw axes through origin
        axins_eig.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        axins_eig.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Draw vertical red line at x=1.0 to indicate critical boundary
        axins_eig.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        
        # Convert eigenvalues to numpy array for easier handling
        eig_array = np.array(nsb_eigenvalue_history)
        epochs_array = np.arange(len(nsb_eigenvalue_history))
        
        # Find the epoch just before crossing the critical boundary (x=1.0)
        # Look for the first epoch where the real part crosses 1.0
        crossing_epoch = None
        for i in range(len(eig_array) - 1):
            if eig_array[i].real < 1.0 and eig_array[i+1].real >= 1.0:
                crossing_epoch = i  # Epoch just before crossing
                break
        
        # Plot trajectory colored by epoch (light blue to dark purple)
        for i in range(len(eig_array) - 1):
            axins_eig.plot(
                [eig_array[i].real, eig_array[i+1].real],
                [eig_array[i].imag, eig_array[i+1].imag],
                '-', color=plt.cm.plasma(i / len(eig_array)), linewidth=1.5, alpha=0.8, zorder=2
            )
        
        # Overlay points colored by epoch
        scatter_eig = axins_eig.scatter(
            eig_array.real, eig_array.imag,
            c=epochs_array,
            cmap='plasma',
            alpha=0.9,
            s=30,
            edgecolors='white',
            linewidths=0.5,
            zorder=3
        )
        
        # Annotate the epoch just before crossing the critical boundary
        if crossing_epoch is not None:
            eig_before_crossing = eig_array[crossing_epoch]
            axins_eig.annotate(
                f'Epoch {crossing_epoch}',
                xy=(eig_before_crossing.real, eig_before_crossing.imag),
                xytext=(15, 15), textcoords='offset points',
                fontsize=8, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red', linewidth=1.0),
                zorder=5,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.7)
            )
        
        # Annotate Epoch 50 (final epoch) - positioned at lower left since it's on the real line
        final_epoch = len(eig_array) - 1
        if final_epoch >= 0:
            eig_final = eig_array[final_epoch]
            axins_eig.annotate(
                f'Epoch {final_epoch}',
                xy=(eig_final.real, eig_final.imag),
                xytext=(-20, -25), textcoords='offset points',  # Lower left position
                fontsize=8, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red', linewidth=1.0),
                zorder=5,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.7)
            )
        
        # Set equal aspect to show the unit circle correctly
        axins_eig.set_aspect('equal', adjustable='box')
        
        # Set axis labels (same fontsize as panel c inset)
        axins_eig.set_xlabel('Re(λ)', fontsize=9, labelpad=2)
        axins_eig.set_ylabel('Im(λ)', fontsize=9, labelpad=2)
        
        # Set ticks to show key values
        axins_eig.set_xticks([-1, 0, 1, 1.5])
        axins_eig.set_yticks([-1, 0, 1])
        axins_eig.tick_params(labelsize=7)
        
        # Set title (same fontsize as panel c inset)
        axins_eig.set_title("NSB's Dominant Eigenvalue", fontsize=10, weight='bold')
        
        # Add colorbar for epochs on the right of the inset
        # Use ax parameter to ensure colorbar is positioned relative to the inset
        cbar_eig = fig.colorbar(scatter_eig, ax=axins_eig, location='right', pad=0.05, shrink=1.0)
        cbar_eig.set_label("Training Epoch", weight='bold', fontsize=9)
        cbar_eig.ax.tick_params(labelsize=6)
        
        # Manually position colorbar to the right of the inset
        # Get inset position in figure coordinates
        pos_inset = axins_eig.get_position()
        # Calculate colorbar position: right edge of inset + small gap
        gap = 0.085  # Gap between inset and colorbar
        cbar_width = 0.02  # Width of colorbar
        cbar_x0 = pos_inset.x1 + gap
        cbar_y0 = pos_inset.y0
        cbar_height = pos_inset.height
        
        # Set colorbar position
        cbar_eig.ax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])

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


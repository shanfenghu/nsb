"""
exp_task_who.py: Duality Validation Dashboard
Visualizes the Neural-Symbolic-Spectral Duality using a trained NSB model.
Panels:
A. Forensic Attribution (Test-set max cluster reconstruction)
B. Spectral Signature (PGF G(s) on complex unit circle)
C. Computational Shield (O(n log n) scaling vs. Baselines)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

# Internal module imports
from nsb.model import NSB
from nsb.task_who import attribute_source, get_prior
from nsb.task_how import compute_how_metrics
from plot_utils import setup_plot_style, save_figure

# REPRODUCIBILITY SEED (matching plot_outbreaktrees.py)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration (matching plot_outbreaktrees.py)
CONFIG = {
    'data_path': Path("data") / "outbreaktrees_sars_mers_counts.csv",
    'test_size': 0.2,
    'output_dir_figures': Path("figures"),
    'nn_params': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 128,
        'hidden_dim': 64,
    }
}

# --------------------------------------------------------------------------
# 1. DATA LOADING AND MODEL TRAINING
# --------------------------------------------------------------------------

def load_and_split_data(path: Path, test_size: float, seed: int):
    """Loads the processed count data and splits it into train/test sets."""
    df = pd.read_csv(path)
    counts = df['offspring_count'].values
    return train_test_split(counts, test_size=test_size, random_state=seed)

def train_nsb_model(train_data: np.ndarray, config: dict) -> NSB:
    """Trains an NSB model on the training data."""
    model = NSB(hidden_dim=config['nn_params']['hidden_dim'])
    model.fit(train_data, epochs=config['nn_params']['epochs'], 
              lr=config['nn_params']['lr'], batch_size=config['nn_params']['batch_size'])
    return model

def lookup_z_true(founder_counts_path: Path, n_test_max: int) -> tuple:
    """
    Looks up the actual founder count (z_true) for the cluster with size n_test_max.
    
    Args:
        founder_counts_path: Path to CSV with outbreak_id, cluster_size_n, founder_count_z
        n_test_max: Cluster size to look up
        
    Returns:
        tuple: (z_true, outbreak_id) or (None, None) if not found
    """
    try:
        df_founders = pd.read_csv(founder_counts_path)
        # Find the outbreak with matching cluster size
        match = df_founders[df_founders['cluster_size_n'] == n_test_max]
        if len(match) > 0:
            z_true = int(match.iloc[0]['founder_count_z'])
            outbreak_id = match.iloc[0]['outbreak_id']
            return z_true, outbreak_id
        else:
            # If exact match not found, return None
            return None, None
    except Exception as e:
        print(f"Warning: Error looking up z_true: {e}")
        return None, None

def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Computes D_KL(P || Q) to quantify information shift.
    
    Args:
        P: Probability distribution (posterior)
        Q: Reference probability distribution (baseline, e.g., flat prior)
        
    Returns:
        KL divergence value
    """
    eps = 1e-10
    # Normalize to ensure they're proper probability distributions
    P_norm = P / (P.sum() + eps)
    Q_norm = Q / (Q.sum() + eps)
    # Compute KL divergence
    kl = np.sum(P_norm * np.log((P_norm + eps) / (Q_norm + eps)))
    return kl

def fit_prior_parameters(founder_counts_path: Path, train_data: np.ndarray, 
                         test_size: float, seed: int) -> dict:
    """
    Fits prior parameters from founder count data.
    
    Args:
        founder_counts_path: Path to CSV with outbreak_id, cluster_size_n, founder_count_z
        train_data: Training cluster sizes (n values) to match outbreaks
        test_size: Test split ratio (to match train/test split)
        seed: Random seed for reproducibility
        
    Returns:
        dict with 'poisson_lambda' and 'negbin_r', 'negbin_p' parameters
    """
    try:
        df_founders = pd.read_csv(founder_counts_path)
        
        # Split founder data the same way as cluster size data
        # We need to match outbreaks by cluster size, so we'll use a similar approach
        # For simplicity, we'll use the same random split on the founder data
        train_founders, _ = train_test_split(
            df_founders['founder_count_z'].values, 
            test_size=test_size, 
            random_state=seed
        )
        
        # Fit Poisson: lambda = mean(z)
        poisson_lambda = float(np.mean(train_founders))
        # Ensure lambda >= 0.1 for numerical stability
        poisson_lambda = max(0.1, poisson_lambda)
        
        # Fit Negative Binomial using method of moments
        # Mean = r(1-p)/p, Variance = r(1-p)/p^2
        # From data: mean_z and var_z
        mean_z = float(np.mean(train_founders))
        var_z = float(np.var(train_founders))
        
        if var_z > mean_z and mean_z > 0:
            # Overdispersed case: use method of moments
            # p = mean / var, r = mean^2 / (var - mean)
            negbin_p = mean_z / var_z
            negbin_r = (mean_z * mean_z) / (var_z - mean_z)
            # Ensure parameters are positive and reasonable
            negbin_p = max(0.1, min(0.9, negbin_p))
            negbin_r = max(0.1, negbin_r)
        else:
            # Underdispersed or mean=0: use defaults
            negbin_p = 0.5
            negbin_r = 2.0
        
        return {
            'poisson_lambda': poisson_lambda,
            'negbin_r': negbin_r,
            'negbin_p': negbin_p
        }
    except FileNotFoundError:
        print(f"Warning: Founder counts file not found at {founder_counts_path}")
        print("Using default prior parameters.")
        return {
            'poisson_lambda': 1.0,
            'negbin_r': 0.5,
            'negbin_p': 0.5
        }
    except Exception as e:
        print(f"Warning: Error fitting prior parameters: {e}")
        print("Using default prior parameters.")
        return {
            'poisson_lambda': 1.0,
            'negbin_r': 0.5,
            'negbin_p': 0.5
        }

# --------------------------------------------------------------------------
# 2. BASELINE DISTRIBUTIONS (For Panel B)
# --------------------------------------------------------------------------

def get_poisson_p_dist(r0: float, k_max: int) -> torch.Tensor:
    """Analytical Poisson offspring law matching NSB R0."""
    k = torch.arange(k_max).float()
    # Poisson PMF: lambda^k * e^(-lambda) / k!
    # For R0 = lambda, we use lambda = R0
    lambda_val = r0
    log_pmf = k * torch.log(torch.tensor(lambda_val)) - torch.tensor(lambda_val) - torch.lgamma(k + 1)
    p_dist = torch.exp(log_pmf)
    return p_dist / p_dist.sum()

def get_nb_p_dist(r0: float, k_max: int, overdispersion: float = 0.1) -> torch.Tensor:
    """Analytical Negative Binomial law matching NSB R0 with overdispersion."""
    # Mean = r(1-p)/p = R0
    # Variance = R0 + R0^2/k_disp
    k_disp = overdispersion
    p = k_disp / (k_disp + r0)
    r = k_disp
    
    k = torch.arange(k_max).float()
    # NB PMF using Gamma formulation
    log_pmf = (torch.lgamma(k + r) - torch.lgamma(k + 1) - torch.lgamma(torch.tensor(r)) +
               r * torch.log(torch.tensor(p)) + k * torch.log(torch.tensor(1 - p)))
    p_dist = torch.exp(log_pmf)
    return p_dist / p_dist.sum()

# --------------------------------------------------------------------------
# 3. SENSITIVITY ANALYSIS (For Panel A Inset)
# --------------------------------------------------------------------------

def scale_offspring_distribution(p_dist: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Scales the offspring distribution to achieve a desired R0 scaling.
    
    Uses a method that preserves the relative shape while scaling the mean.
    For scale_factor > 1, increases transmission; for < 1, decreases it.
    
    Args:
        p_dist: Original offspring distribution
        scale_factor: Multiplier for R0 (1.0 = no change, 0.8 = 20% reduction, 1.2 = 20% increase)
        
    Returns:
        Scaled offspring distribution with R0' = scale_factor * R0
    """
    device = p_dist.device
    p_np = p_dist.numpy() if torch.is_tensor(p_dist) else p_dist
    
    # Compute current R0
    k_range = np.arange(len(p_np))
    r0_current = np.sum(k_range * p_np)
    r0_target = scale_factor * r0_current
    
    if abs(scale_factor - 1.0) < 1e-6:
        return p_dist if torch.is_tensor(p_dist) else torch.from_numpy(p_np).float()
    
    # Method: Use exponential tilting to shift the distribution
    # This preserves the shape while scaling the mean
    # We adjust probabilities: p_k' ∝ p_k * exp(λ * k) where λ is chosen to achieve target R0
    
    # Find λ using binary search or Newton's method
    # For simplicity, use a linear approximation: adjust probabilities proportionally to k
    # More sophisticated: use exponential tilting
    
    # Simple approach: weight by k^α where α is chosen to achieve target R0
    # This is a simplified version - in practice, exponential tilting would be more accurate
    
    if r0_target > 0 and r0_target != r0_current:
        # Use exponential tilting: p_k' ∝ p_k * exp(λ * k)
        # Find λ using binary search to achieve target R0
        
        # Binary search for λ
        lambda_low = -2.0
        lambda_high = 2.0
        max_iter = 30
        tolerance = 1e-4
        
        for _ in range(max_iter):
            lambda_mid = (lambda_low + lambda_high) / 2.0
            
            # Apply exponential tilting
            weights = np.exp(lambda_mid * k_range)
            p_tilted = p_np * weights
            p_tilted = p_tilted / (p_tilted.sum() + 1e-10)
            
            r0_tilted = np.sum(k_range * p_tilted)
            
            if abs(r0_tilted - r0_target) / r0_target < tolerance:
                p_scaled = p_tilted
                break
            elif r0_tilted < r0_target:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
        else:
            # If binary search didn't converge, use the last result
            p_scaled = p_tilted
    else:
        p_scaled = p_np
    
    return torch.from_numpy(p_scaled).float().to(device)

# --------------------------------------------------------------------------
# 4. COMPLEXITY BENCHMARKING (For Panel C)
# --------------------------------------------------------------------------

def run_benchmarks(n_range, k_depth, p_dist):
    """Measures NSB runtime and projects theoretical baselines."""
    nsb_times = []
    
    for n in n_range:
        start = time.perf_counter()
        _ = attribute_source(p_dist, n, prior_type="flat")
        nsb_times.append(time.perf_counter() - start)
    
    # Project theoretical baselines from first measurement
    # Tree Search: O(4^n) normalized to first n
    n0 = n_range[0]
    tree_search = [nsb_times[0] * (4**(n - n0)) for n in n_range]
    # Direct Conv: O(n^2) normalized to first n
    direct_conv = [nsb_times[0] * ((n / n0)**2) for n in n_range]
    
    return nsb_times, tree_search, direct_conv

# --------------------------------------------------------------------------
# 4. DASHBOARD GENERATION
# --------------------------------------------------------------------------

def plot_duality_dashboard(p_nsb: torch.Tensor, n_test_max: int, z_true: int, 
                          pathogen_name: str = "SARS/MERS", test_data: np.ndarray = None,
                          prior_params: dict = None, outbreak_id: str = None):
    """Creates the Duality Validation Dashboard with three panels."""
    setup_plot_style()
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], 
                          height_ratios=[1, 1], hspace=0.3, wspace=0.15)

    # --- PANEL A: Forensic "Cold Case" Reconstruction ---
    ax_a = fig.add_subplot(gs[:, 0])
    
    # Calculate K (distribution length) and n (cluster size)
    K = len(p_nsb)  # Length of learned offspring distribution
    n = n_test_max   # Maximum cluster size from test data
    
    # Determine visible x-axis range
    active_limit = min(n_test_max, 40)
    
    # Calculate Posterior Surface for different priors
    z_range = np.arange(1, n_test_max + 1)
    
    # Raw Likelihood (using flat prior to get unnormalized likelihood surface)
    likelihood_surface = attribute_source(p_nsb, n_test_max, prior_type="flat")
    likelihood_np = likelihood_surface.numpy()
    
    # Archetype 1: Clinical (Flat)
    post_flat = likelihood_np.copy()
    # Archetype 2: Community (Poisson)
    if prior_params is None:
        poisson_lambda = 1.0
    else:
        poisson_lambda = prior_params.get('poisson_lambda', 1.0)
    post_comm = attribute_source(p_nsb, n_test_max, prior_type="community", 
                                 prior_params={"lambda": poisson_lambda}).numpy()
    # Archetype 3: Clustered (NegBin)
    if prior_params is None:
        negbin_r, negbin_p = 0.5, 0.5
    else:
        negbin_r = prior_params.get('negbin_r', 0.5)
        negbin_p = prior_params.get('negbin_p', 0.5)
    post_clus = attribute_source(p_nsb, n_test_max, prior_type="clustered", 
                                 prior_params={"r": negbin_r, "p": negbin_p}).numpy()

    # Visual Layout A: Gradient "Likelihood Mountain" background
    # Use a simpler, more visible gradient approach
    # Create a single fill with a gradient colormap applied vertically
    colors_gradient = ['#E0F7FA', '#80DEEA', '#26C6DA', '#00ACC1']
    cmap = LinearSegmentedColormap.from_list('likelihood_gradient', colors_gradient, N=256)
    
    # Create a meshgrid for gradient effect
    # Use fewer layers but with better visibility
    n_layers = 15
    for i in range(n_layers):
        # Progress from light to dark (top should be darker)
        color_val = cmap(i / (n_layers - 1) if n_layers > 1 else 0)
        # Use higher alpha for better visibility
        alpha_val = 0.25 + 0.20 * (i / n_layers)  # Range from 0.25 to 0.45
        # Fill from bottom to a fraction of the likelihood
        height_frac = (i + 1) / n_layers
        ax_a.fill_between(z_range, 0, likelihood_np * height_frac, 
                         color=color_val, alpha=alpha_val, zorder=0)
    
    # Create a visible legend entry using a Patch
    # We'll manually add this to the legend later
    spectral_patch = Patch(facecolor='#26C6DA', edgecolor='#00ACC1', alpha=0.6, 
                          label=r"NSB's Learned Likelihood $P(C=n \mid Z=z; \theta)$")
    
    # Solid line for Flat Prior
    ax_a.plot(z_range, post_flat, color='#2E86AB', ls='-', lw=3, 
             label="Posterior (Flat Prior): Uniform Risk of Seeding", zorder=3)
    
    # Smooth curves for Poisson and NegBin priors
    ax_a.plot(z_range, post_comm, color='#06A77D', ls='-', lw=2.5, 
             label="Posterior (Poisson Prior): Sparse, Independent Seeding", zorder=4)
    ax_a.plot(z_range, post_clus, color='#A23B72', ls='-', lw=2.5, 
             label="Posterior (NegBin Prior): Seeding in Bursts or Clusters", zorder=4)
    
    # Calculate MAP (Maximum A Posteriori) for each posterior
    map_flat = z_range[np.argmax(post_flat)]
    map_comm = z_range[np.argmax(post_comm)]
    map_clus = z_range[np.argmax(post_clus)]
    
    # Get y-axis limits for annotation positioning
    y_max = max(post_flat.max(), post_comm.max(), post_clus.max())
    
    # Add MAP vertical lines with annotations
    ax_a.axvline(x=map_flat, color='#2E86AB', ls='--', lw=2, alpha=0.7, zorder=5)
    ax_a.text(map_flat, y_max * 0.8, f'$z={map_flat}$', 
              color='#2E86AB', fontsize=12, ha='center', va='top', 
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#2E86AB'))
    
    ax_a.axvline(x=map_comm, color='#06A77D', ls='--', lw=2, alpha=0.7, zorder=5)
    ax_a.text(map_comm, y_max * 0.85, f'$z={map_comm}$', 
              color='#06A77D', fontsize=12, ha='center', va='top', 
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#06A77D'))
    
    ax_a.axvline(x=map_clus, color='#A23B72', ls='--', lw=2, alpha=0.7, zorder=5)
    ax_a.text(map_clus, y_max * 0.75, f'$z={map_clus}$', 
              color='#A23B72', fontsize=12, ha='center', va='top', 
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#A23B72'))
    
    # ML Ground Truth Marker
    ax_a.axvline(x=z_true, color='#FF6B35', ls='--', lw=2.5, alpha=0.9, 
                label=f"Ground Truth ($z={z_true}$)", zorder=6)
    
    # Add KL Divergence Inset (middle left, side by side with heatmap)
    ax_inset = ax_a.inset_axes([0.2, 0.35, 0.3, 0.3])  # [x, y, width, height]
    
    # Compute priors for each distribution type
    prior_flat = get_prior(n_test_max, prior_type="flat").numpy()
    
    if prior_params is None:
        poisson_lambda = 1.0
        negbin_r, negbin_p = 0.5, 0.5
    else:
        poisson_lambda = prior_params.get('poisson_lambda', 1.0)
        negbin_r = prior_params.get('negbin_r', 0.5)
        negbin_p = prior_params.get('negbin_p', 0.5)
    
    prior_comm = get_prior(n_test_max, prior_type="community", 
                           params={"lambda": poisson_lambda}).numpy()
    prior_clus = get_prior(n_test_max, prior_type="clustered", 
                          params={"r": negbin_r, "p": negbin_p}).numpy()
    
    # Compute KL divergences: D_KL(Posterior || Prior)
    # This measures the information gain from observations (likelihood)
    kl_flat = compute_kl_divergence(post_flat, prior_flat)
    kl_comm = compute_kl_divergence(post_comm, prior_comm)
    kl_clus = compute_kl_divergence(post_clus, prior_clus)
    
    # Create bar chart using numeric positions - show all three
    x_pos = np.array([0, 1, 2])
    bars = ax_inset.bar(x_pos, [kl_flat, kl_comm, kl_clus], 
                       color=['#2E86AB', '#06A77D', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=1)
    ax_inset.set_xticks(x_pos)
    # Use shortened but descriptive labels that match the main legend style
    ax_inset.set_xticklabels(['Flat', 'Poisson', 'NegBin'], fontsize=12)
    ax_inset.set_title("KL(Posterior $\\|$ Prior)", fontsize=12, fontweight='bold')
    ax_inset.set_ylabel("Info. Gain (KL)", fontsize=12)
    ax_inset.tick_params(labelsize=12)
    ax_inset.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, [kl_flat, kl_comm, kl_clus]):
        height = bar.get_height()
        if height > 0.001:  # Only show label if value is significant
            ax_inset.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=12)
    
    # Add Inversion Sensitivity Heatmap Inset (middle right, side by side with KL inset)
    ax_heatmap = ax_a.inset_axes([0.6, 0.35, 0.3, 0.3])  # [x, y, width, height]
    
    # Define scaling factor range (R0 multiplier)
    scale_range = np.linspace(0.8, 1.2, 25)  # 25 points from 80% to 120% of learned R0
    z_range_heatmap = np.arange(1, active_limit + 1)  # Match the main plot's active zone
    
    # Compute likelihood surface for each scale factor
    heatmap_data = np.zeros((len(scale_range), len(z_range_heatmap)))
    
    print("   Computing sensitivity heatmap...")
    for i, scale in enumerate(scale_range):
        # Scale the offspring distribution
        p_scaled = scale_offspring_distribution(p_nsb, scale)
        
        # Compute likelihood surface for this scaled distribution
        # Use the maximum cluster size for the heatmap
        likelihood_scaled = attribute_source(p_scaled, n_test_max, prior_type="flat")
        likelihood_scaled_np = likelihood_scaled.numpy()
        
        # Extract the relevant z range
        z_indices = z_range_heatmap - 1  # Convert to 0-based indexing
        z_indices = z_indices[z_indices < len(likelihood_scaled_np)]
        if len(z_indices) > 0:
            heatmap_data[i, :len(z_indices)] = likelihood_scaled_np[z_indices]
    
    # Create heatmap
    im = ax_heatmap.imshow(heatmap_data, aspect='auto', origin='lower', 
                          cmap='YlOrRd', interpolation='bilinear')
    
    # Set ticks and labels
    z_ticks = np.linspace(0, len(z_range_heatmap)-1, min(6, len(z_range_heatmap)), dtype=int)
    ax_heatmap.set_xticks(z_ticks)
    ax_heatmap.set_xticklabels([str(z_range_heatmap[t]) for t in z_ticks], fontsize=12)
    
    scale_ticks = np.linspace(0, len(scale_range)-1, 5, dtype=int)
    ax_heatmap.set_yticks(scale_ticks)
    ax_heatmap.set_yticklabels([f'{scale_range[t]:.2f}' for t in scale_ticks], fontsize=12)
    
    ax_heatmap.set_xlabel("Founders ($z$)", fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel("$R_0$ Scale", fontsize=12, fontweight='bold')
    ax_heatmap.set_title("Inversion Sensitivity", fontsize=12, fontweight='bold')
    ax_heatmap.tick_params(labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Likelihood", fontsize=12, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Zoom to active zone (likely 1-30 or 1-40)
    ax_a.set_xlim(0.5, active_limit + 0.5)
    
    # Build title with optional outbreak metadata
    title_base = f'(a) Task "Who": {pathogen_name} ($n_\\max={n}$, $K={K}$)'
    if outbreak_id is not None:
        # Extract readable info from outbreak_id (e.g., "sau.2019.mers.1.00" -> "SAU 2019 MERS")
        parts = outbreak_id.split('.')
        if len(parts) >= 3:
            country = parts[0].upper()
            year = parts[1]
            pathogen_short = parts[2].upper()
            title_base += f' [{country} {year} {pathogen_short}]'
    ax_a.set_title(title_base, loc='center', fontsize=14, fontweight='bold')
    ax_a.set_xlabel("Potential Founders / Patient Zeros ($Z=z$)", fontsize=12, fontweight='bold')
    ax_a.set_ylabel("Posterior Probability $P(Z=z|C=n; \\theta)$", fontsize=12, fontweight='bold')
    
    # Create legend with all elements including the spectral evidence patch
    handles, labels = ax_a.get_legend_handles_labels()
    # Add the spectral evidence patch to the beginning of the legend
    handles.insert(0, spectral_patch)
    labels.insert(0, spectral_patch.get_label())
    ax_a.legend(handles, labels, loc='upper right', frameon=True, fontsize=12)
    ax_a.grid(alpha=0.3, linestyle='--')

    # --- PANEL B: Spectral DNA (G(s) on Unit Circle) ---
    ax_b = fig.add_subplot(gs[0, 1])
    
    metrics = compute_how_metrics(p_nsb)
    r0 = metrics['r0']
    k_max = len(p_nsb)
    
    # Align baselines with same R0
    p_pois = get_poisson_p_dist(r0, k_max)
    p_nb = get_nb_p_dist(r0, k_max, overdispersion=0.1)
    
    # Evaluation on complex unit circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    s = np.exp(1j * theta)
    k_idx = np.arange(k_max)
    
    def get_pgf_trace(p):
        """Compute G(s) = sum(p_k * s^k) for s on unit circle."""
        p_np = p.numpy() if torch.is_tensor(p) else p
        # Broadcast: p_np is (k_max,), s is (1000,), k_idx is (k_max,)
        # We want: sum_k p_k * s^k for each s
        s_powers = np.power(s[None, :], k_idx[:, None])  # Shape: (k_max, 1000)
        return np.dot(p_np, s_powers)  # Shape: (1000,)

    g_nsb = get_pgf_trace(p_nsb)
    g_pois = get_pgf_trace(p_pois)
    g_nb = get_pgf_trace(p_nb)

    # Visual Layout B: Conformal Zoom with Phase-Amplitude Color Mapping
    # Unit Shield as faint backdrop (full circle)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='lightgray', linestyle='--', 
                            linewidth=1, alpha=0.2, zorder=0)
    ax_b.add_patch(unit_circle)
    
    # Zoom into critical area around G(1)=1
    ax_b.set_xlim(0.5, 1.1)
    ax_b.set_ylim(-0.3, 0.3)
    
    # Vertical dashed line at x=1 to indicate unit circle boundary
    ax_b.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Phase-Amplitude Color Mapping: Color NSB trace by angle θ
    # Normalize theta to [0, 1] for colormap
    cmap_phase = plt.cm.viridis  # Use viridis colormap for phase encoding
    
    # Plot NSB with color-by-phase
    for i in range(len(theta) - 1):
        ax_b.plot([g_nsb.real[i], g_nsb.real[i+1]], 
                 [g_nsb.imag[i], g_nsb.imag[i+1]], 
                 color=cmap_phase(theta[i] / (2 * np.pi)), 
                 lw=3.5, alpha=0.9, zorder=4)
    
    # Add glow effect for NSB (subtle shadow)
    for offset in [0.002, 0.004]:
        ax_b.plot(g_nsb.real + offset, g_nsb.imag + offset, 
                 color='#1f77b4', lw=2, alpha=0.15, zorder=3)
    
    # Plot baselines with same colors as panel (a)
    ax_b.plot(g_pois.real, g_pois.imag, color='#06A77D', ls='--', lw=3, alpha=0.7, 
             label="Poisson Baseline", zorder=2)
    ax_b.plot(g_nb.real, g_nb.imag, color='#A23B72', ls=':', lw=3, alpha=0.7, 
             label="NegBin Baseline", zorder=2)
    
    # Residual Shading: Area between Poisson and NSB
    # Find points where both are in the visible range
    mask = (g_nsb.real >= 0.5) & (g_nsb.real <= 1.1) & (g_nsb.imag >= -0.3) & (g_nsb.imag <= 0.3)
    mask_pois = (g_pois.real >= 0.5) & (g_pois.real <= 1.1) & (g_pois.imag >= -0.3) & (g_pois.imag <= 0.3)
    
    # Interpolate to same length for filling
    if np.sum(mask) > 10 and np.sum(mask_pois) > 10:
        # Sample points for filling
        n_fill = min(200, len(g_nsb))
        indices = np.linspace(0, len(g_nsb)-1, n_fill, dtype=int)
        indices_pois = np.linspace(0, len(g_pois)-1, n_fill, dtype=int)
        
        g_nsb_fill = g_nsb[indices]
        g_pois_fill = g_pois[indices_pois]
        
        # Create polygon for shading
        fill_x = np.concatenate([g_nsb_fill.real, g_pois_fill.real[::-1]])
        fill_y = np.concatenate([g_nsb_fill.imag, g_pois_fill.imag[::-1]])
        ax_b.fill(fill_x, fill_y, color='orange', alpha=0.15, zorder=1, 
                 label="Overdispersion Gap")
    
    # Vector Needles: Show shift from Poisson to NSB at selected points
    n_arrows = 8
    arrow_indices = np.linspace(0, len(g_nsb)-1, n_arrows, dtype=int)
    arrow_indices_pois = np.linspace(0, len(g_pois)-1, n_arrows, dtype=int)
    
    for idx, idx_pois in zip(arrow_indices, arrow_indices_pois):
        if idx < len(g_nsb) and idx_pois < len(g_pois):
            x_start = g_pois.real[idx_pois]
            y_start = g_pois.imag[idx_pois]
            x_end = g_nsb.real[idx]
            y_end = g_nsb.imag[idx]
            
            # Only draw if both points are in visible range
            if (0.5 <= x_start <= 1.1 and -0.3 <= y_start <= 0.3 and
                0.5 <= x_end <= 1.1 and -0.3 <= y_end <= 0.3):
                dx = x_end - x_start
                dy = y_end - y_start
                # Only draw if arrow is significant
                if np.sqrt(dx**2 + dy**2) > 0.01:
                    ax_b.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                                arrowprops=dict(arrowstyle='->', lw=1.5, 
                                              color='orange', alpha=0.4, zorder=3))
    
    # Red anchor point at G(1)=1+0i
    ax_b.scatter([1], [0], color='red', s=100, zorder=6, marker='o', 
                edgecolors='black', linewidths=2, label="$G(1)=1 \\Leftrightarrow \\sum p_k = 1$")
    
    # Create a proxy artist for NSB with phase coloring in legend
    nsb_proxy = Line2D([0], [0], color='#1f77b4', lw=3.5, label="NSB's Neural PGF")
    
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.set_title(f"(b) Spectral Signature of {pathogen_name}", 
                   loc='center', fontsize=12, fontweight='bold')
    ax_b.set_xlabel('Re($G(s)$)', fontsize=12, fontweight='bold')
    ax_b.set_ylabel('Im($G(s)$)', fontsize=12, fontweight='bold')
    
    # Update legend to include NSB proxy and residual shading
    handles, labels = ax_b.get_legend_handles_labels()
    # Replace any existing NSB entry with our proxy
    if "NSB's Neural PGF" in labels:
        idx = labels.index("NSB's Neural PGF")
        handles[idx] = nsb_proxy
    else:
        handles.insert(0, nsb_proxy)
        labels.insert(0, "NSB's Neural PGF")
    
    ax_b.legend(handles, labels, loc='lower left', fontsize=10, frameon=True)
    ax_b.grid(alpha=0.2, linestyle='--')

    # --- PANEL C: Computational "Forensic Shield" ---
    ax_c = fig.add_subplot(gs[1, 1])
    
    n_range = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000])
    nsb_t, tree_t, conv_t = run_benchmarks(n_range, k_max, p_nsb)
    
    # Convert to numpy arrays for vectorized operations
    nsb_t = np.array(nsb_t)
    tree_t = np.array(tree_t)
    conv_t = np.array(conv_t)
    
    # Theoretical O(nK log(nK)) reference line
    n0 = n_range[0]
    nk_log_nk = [nsb_t[0] * ((n * k_max * np.log(n * k_max)) / (n0 * k_max * np.log(n0 * k_max))) 
                 for n in n_range]
    
    # Determine y_max first
    y_max = min(max(nsb_t) * 10, 1000)
    
    # Filter tree search to only plot points within visible range to avoid vertical line
    tree_mask = tree_t <= y_max * 0.9
    n_range_tree = n_range[tree_mask]
    tree_t_visible = tree_t[tree_mask]
    
    # Visual Layout C
    ax_c.loglog(n_range, nsb_t, 'o-', color='#2ca02c', lw=3, markersize=8, 
               label="NSB One-Pass (Measured)", zorder=3)
    ax_c.loglog(n_range, nk_log_nk, '--', color='black', lw=2, alpha=0.7, 
               label=r"NSB Theoretical $O(nK \log(nK))$", zorder=2)
    ax_c.loglog(n_range, conv_t, '--', color='#ff7f0e', lw=2.5, alpha=0.8, 
               label=r"Direct (Time Domain) Conv $O((nK)^2)$", zorder=2)
    
    # Only plot tree search values that are within the visible range
    if len(n_range_tree) > 0:
        ax_c.loglog(n_range_tree, tree_t_visible, '--', color='#d62728', lw=2.5, alpha=0.8, 
                   label="Exponential Tree Search $O(4^n)$", zorder=2)
    
    # Set limits
    ax_c.set_ylim(min(nsb_t) * 0.3, y_max)
    ax_c.set_xlim(8, n_range[-1] * 1.2)
    
    ax_c.set_title("(c) Time Complexity and Scaling", 
                   loc='center', fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Cluster Size ($n$)", fontsize=12, fontweight='bold')
    ax_c.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax_c.legend(loc='lower right', fontsize=10, frameon=True)
    ax_c.grid(True, which="both", ls="-", alpha=0.2)

    save_figure(fig, "exp_task_who")
    print(f"Dashboard saved to '{CONFIG['output_dir_figures'] / 'exp_task_who.pdf'}'")
    plt.close()

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Duality Validation Dashboard: Forensic Cold Case Reconstruction")
    print("=" * 70)
    
    # Load and split data (matching plot_outbreaktrees.py)
    print(f"\n1. Loading data from '{CONFIG['data_path']}'...")
    train_data, test_data = load_and_split_data(CONFIG['data_path'], 
                                                 CONFIG['test_size'], SEED)
    print(f"   Train set: {len(train_data)} samples")
    print(f"   Test set: {len(test_data)} samples")
    
    # Train NSB model
    print(f"\n2. Training NSB model (hidden_dim={CONFIG['nn_params']['hidden_dim']})...")
    model = train_nsb_model(train_data, CONFIG)
    print("   Training complete.")
    
    # Get learned distribution
    print(f"\n3. Extracting learned offspring distribution...")
    p_learned = model.predict_pmf(k_max=150)  # Match k_max from CONFIG
    p_learned = torch.from_numpy(p_learned).float()
    p_learned = p_learned / p_learned.sum()  # Ensure normalization
    
    metrics = compute_how_metrics(p_learned)
    print(f"   Learned R0: {metrics['r0']:.4f}")
    print(f"   Distribution support: k_max = {len(p_learned)}")
    
    # Find largest cluster in test set
    print(f"\n4. Finding largest cluster in test set...")
    n_test_max = int(test_data.max())
    print(f"   Maximum cluster size: n = {n_test_max}")
    
    # Look up actual z_true from founder counts metadata
    print(f"\n5. Looking up ground truth founder count...")
    founder_counts_path = Path("data") / "outbreaktrees_founder_counts.csv"
    z_true, outbreak_id = lookup_z_true(founder_counts_path, n_test_max)
    
    if z_true is not None:
        print(f"   Found: z_true = {z_true} (outbreak: {outbreak_id})")
    else:
        # Fallback: use reasonable estimate if lookup fails
        z_true = max(1, int(n_test_max * 0.02))
        print(f"   Warning: Could not find exact match, using estimate: z_true = {z_true}")
    
    # Fit prior parameters from founder count data
    print(f"\n6. Fitting prior parameters from founder count data...")
    prior_params = fit_prior_parameters(founder_counts_path, train_data, 
                                       CONFIG['test_size'], SEED)
    print(f"   Poisson lambda: {prior_params['poisson_lambda']:.3f}")
    print(f"   NegBin r: {prior_params['negbin_r']:.3f}, p: {prior_params['negbin_p']:.3f}")
    
    # Determine pathogen name - using combined SARS/MERS dataset
    pathogen_name = "SARS/MERS"  # Combined dataset as in plot_outbreaktrees.py
    
    # Generate dashboard
    print(f"\n6. Generating Duality Validation Dashboard...")
    plot_duality_dashboard(p_learned, n_test_max, z_true, pathogen_name, 
                          test_data=test_data, prior_params=prior_params, 
                          outbreak_id=outbreak_id)
    
    print("\n" + "=" * 70)
    print("Dashboard generation complete!")
    print("=" * 70)


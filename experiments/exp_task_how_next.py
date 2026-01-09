"""
exp_task_how_next.py: Forensic Forecasting Dashboard
Analyzes Task "How" (Entropy) and Task "Next" (Extinction/Volatility).
Panels:
A. Extinction Phase Transition: E(R0) fixed-point stability analysis
B. Forensic Entropy: Information loss H(z|n) vs. cluster size n
C. Branching Volatility: Tail risk and "Bimodal Tension" (Extinction vs. Superspreading)
D. Spectral Energy Dissipation: Recursive G_m(s) decay on the unit circle
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.optimize import fsolve
from scipy.stats import entropy as scipy_entropy
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

# Internal module imports
from nsb.model import NSB
from nsb.task_who import attribute_source, get_prior
from nsb.task_how import compute_how_metrics, _solve_extinction_newton
from plot_utils import setup_plot_style, save_figure

# REPRODUCIBILITY SEED (matching exp_task_who.py)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration (matching exp_task_who.py)
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
# 1. DATA LOADING AND MODEL TRAINING (Reuse from exp_task_who.py)
# --------------------------------------------------------------------------

def load_and_split_data(path: Path, test_size: float, seed: int):
    """Loads cluster size data and splits into train/test."""
    df = pd.read_csv(path)
    data = df['offspring_count'].values.astype(float)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    return train_data, test_data

def train_nsb_model(train_data: np.ndarray, config: dict) -> NSB:
    """Trains an NSB model on the training data."""
    model = NSB(hidden_dim=config['nn_params']['hidden_dim'])
    model.fit(train_data, epochs=config['nn_params']['epochs'], 
              lr=config['nn_params']['lr'], batch_size=config['nn_params']['batch_size'])
    return model

# --------------------------------------------------------------------------
# 2. BASELINE DISTRIBUTIONS (Reuse from exp_task_who.py)
# --------------------------------------------------------------------------

def get_poisson_p_dist(r0: float, k_max: int) -> torch.Tensor:
    """Analytical Poisson offspring law matching NSB R0."""
    k = torch.arange(k_max).float()
    lambda_val = r0
    log_pmf = k * torch.log(torch.tensor(lambda_val)) - torch.tensor(lambda_val) - torch.lgamma(k + 1)
    p_dist = torch.exp(log_pmf)
    return p_dist / p_dist.sum()

def get_nb_p_dist(r0: float, k_max: int, overdispersion: float = 0.1) -> torch.Tensor:
    """Analytical Negative Binomial law matching NSB R0 with overdispersion."""
    k_disp = overdispersion
    p = k_disp / (k_disp + r0)
    r = k_disp
    
    k = torch.arange(k_max).float()
    log_pmf = (torch.lgamma(k + r) - torch.lgamma(k + 1) - torch.lgamma(torch.tensor(r)) +
               r * torch.log(torch.tensor(p)) + k * torch.log(torch.tensor(1 - p)))
    p_dist = torch.exp(log_pmf)
    return p_dist / p_dist.sum()

# --------------------------------------------------------------------------
# 3. EXTINCTION PROBABILITY COMPUTATION
# --------------------------------------------------------------------------

def compute_extinction_probability(p_dist: torch.Tensor) -> float:
    """
    Computes extinction probability by solving s = G(s) for s in [0, 1].
    Uses the Newton-Raphson method from task_how.py for consistency.
    """
    metrics = compute_how_metrics(p_dist)
    return metrics['extinction_prob']

def scale_offspring_distribution(p_dist: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Scales the offspring distribution to achieve a desired R0 scaling.
    Reuses the exponential tilting method from exp_task_who.py.
    """
    device = p_dist.device
    p_np = p_dist.numpy() if torch.is_tensor(p_dist) else p_dist
    
    k_range = np.arange(len(p_np))
    r0_current = np.sum(k_range * p_np)
    r0_target = scale_factor * r0_current
    
    if abs(scale_factor - 1.0) < 1e-6:
        return p_dist if torch.is_tensor(p_dist) else torch.from_numpy(p_np).float()
    
    if r0_target > 0 and r0_target != r0_current:
        # Binary search for exponential tilting parameter
        lambda_low = -2.0
        lambda_high = 2.0
        max_iter = 30
        tolerance = 1e-4
        
        for _ in range(max_iter):
            lambda_mid = (lambda_low + lambda_high) / 2.0
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
            p_scaled = p_tilted
    else:
        p_scaled = p_np
    
    return torch.from_numpy(p_scaled).float().to(device)

# --------------------------------------------------------------------------
# 4. GENERATION PGF COMPUTATION
# --------------------------------------------------------------------------

def compute_generation_pgf(p_dist: torch.Tensor, m: int, s_points: np.ndarray) -> np.ndarray:
    """
    Computes the m-th generation PGF: G_m(s) = G(G_{m-1}(s)).
    
    Args:
        p_dist: Offspring distribution
        m: Generation number
        s_points: Complex points on unit circle to evaluate
        
    Returns:
        Array of G_m(s) values
    """
    p_np = p_dist.numpy() if torch.is_tensor(p_dist) else p_dist
    k_idx = np.arange(len(p_np))
    
    # Start with s_points
    g_m = s_points.copy()
    
    # Recursively apply G: G_m(s) = G(G_{m-1}(s))
    for generation in range(m):
        # G(s) = sum_k p_k * s^k
        # Broadcast: g_m is (n_points,), k_idx is (k_max,)
        # We want: sum_k p_k * g_m^k for each g_m
        g_m_powers = np.power(g_m[:, None], k_idx[None, :])  # Shape: (n_points, k_max)
        g_m = np.dot(g_m_powers, p_np)  # Shape: (n_points,)
    
    return g_m

# --------------------------------------------------------------------------
# 5. MAIN DASHBOARD FUNCTION
# --------------------------------------------------------------------------

def plot_forecasting_dashboard(p_nsb: torch.Tensor, pathogen_name: str = "SARS/MERS"):
    """
    Creates the 2x2 Forensic Forecasting Dashboard.
    
    Panels:
    A. Extinction Phase Transition E(R0)
    B. Forensic Entropy H(z|n)
    C. Branching Volatility (Tail Risk)
    D. Spectral Energy Dissipation
    """
    setup_plot_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics = compute_how_metrics(p_nsb)
    r0_learned = metrics['r0']
    k_max = len(p_nsb)
    
    # --- PANEL A: Extinction Phase Transition ---
    ax_a = fig.add_subplot(gs[0, 0])
    
    print("   Computing extinction phase transition...")
    r0_range = np.linspace(0.5, 2.0, 100)
    ext_nsb = []
    ext_pois = []
    ext_nb = []
    
    for r0_target in r0_range:
        # Scale NSB distribution to achieve target R0
        scale_factor = r0_target / r0_learned
        p_scaled = scale_offspring_distribution(p_nsb, scale_factor)
        ext_nsb.append(compute_extinction_probability(p_scaled))
        
        # Poisson extinction: For Poisson, E = 1 if R0 <= 1, else solve s = exp(R0(s-1))
        if r0_target <= 1.0:
            ext_pois.append(1.0)
        else:
            # Solve s = exp(R0(s-1)) for s < 1
            def pois_fixed_point(s):
                return np.exp(r0_target * (s - 1)) - s
            s_star = fsolve(pois_fixed_point, 0.5)[0]
            ext_pois.append(min(1.0, float(s_star)))
        
        # NegBin extinction (using same method as NSB)
        p_nb = get_nb_p_dist(r0_target, k_max, overdispersion=0.1)
        ext_nb.append(compute_extinction_probability(p_nb))
    
    # Visual Layout A
    ax_a.plot(r0_range, ext_nsb, color='#1f77b4', lw=3, label="NSB's Learned Law", zorder=3)
    ax_a.plot(r0_range, ext_pois, color='#06A77D', ls='--', lw=2.5, alpha=0.8, 
             label="Poisson Baseline", zorder=2)
    ax_a.plot(r0_range, ext_nb, color='#A23B72', ls=':', lw=2.5, alpha=0.8, 
             label="NegBin Baseline", zorder=2)
    
    # Critical point marker
    ax_a.axvline(1.0, color='red', ls='--', lw=2, alpha=0.6, zorder=1, label="Critical Point ($R_0=1$)")
    
    # Highlight learned R0
    ext_learned = compute_extinction_probability(p_nsb)
    ax_a.scatter([r0_learned], [ext_learned], color='#FF6B35', s=150, zorder=5, 
                marker='*', edgecolors='black', linewidths=1.5, 
                label=f"Learned ($R_0={r0_learned:.3f}$)")
    
    # Shade subcritical region
    ax_a.axvspan(0.5, 1.0, alpha=0.1, color='red', zorder=0, label="Subcritical Zone")
    
    ax_a.set_xlabel("Reproductive Number ($R_0=G'(1)$)", fontsize=12, fontweight='bold')
    ax_a.set_ylabel("Extinction Probability $q=G(q)$", fontsize=12, fontweight='bold')
    ax_a.set_title("(a) Outbreak Survival Thresholds (SARS/MERS)", 
                   fontsize=14, fontweight='bold')
    ax_a.legend(loc='lower right', fontsize=10, frameon=True)
    ax_a.grid(alpha=0.3, linestyle='--')
    ax_a.set_xlim(0.5, 2.0)
    ax_a.set_ylim(0, 1.05)
    
    # Add inset to zoom into R0 ∈ [0.9, 1.1]
    ax_inset = ax_a.inset_axes([0.62, 0.45, 0.35, 0.35])  # [x, y, width, height]
    
    # Filter data for zoom region
    mask = (r0_range >= 0.9) & (r0_range <= 1.1)
    r0_zoom = r0_range[mask]
    ext_nsb_zoom = np.array(ext_nsb)[mask]
    ext_pois_zoom = np.array(ext_pois)[mask]
    ext_nb_zoom = np.array(ext_nb)[mask]
    
    # Plot zoomed region (only NSB and NegBin, no Poisson for clarity)
    ax_inset.plot(r0_zoom, ext_nsb_zoom, color='#1f77b4', lw=2.5, label="NSB", zorder=3)
    ax_inset.plot(r0_zoom, ext_nb_zoom, color='#A23B72', ls=':', lw=2, alpha=0.8, 
                 label="NegBin", zorder=2)
    
    # Critical point marker
    ax_inset.axvline(1.0, color='red', ls='--', lw=1.5, alpha=0.6, zorder=1)
    
    # Highlight learned R0
    if 0.9 <= r0_learned <= 1.1:
        ext_learned = compute_extinction_probability(p_nsb)
        ax_inset.scatter([r0_learned], [ext_learned], color='#FF6B35', s=100, zorder=5, 
                        marker='*', edgecolors='black', linewidths=1)
    
    ax_inset.set_xlim(0.9, 1.1)
    ax_inset.set_ylim(min(min(ext_nsb_zoom), min(ext_nb_zoom)) * 0.98, 
                      max(max(ext_nsb_zoom), max(ext_nb_zoom)) * 1.02)
    ax_inset.set_xlabel("$R_0$", fontsize=10, fontweight='bold')
    ax_inset.set_ylabel("$q$", fontsize=10, fontweight='bold')
    ax_inset.set_title("Zoom In", fontsize=10, fontweight='bold')
    ax_inset.tick_params(labelsize=9)
    ax_inset.grid(alpha=0.3, linestyle='--')
    ax_inset.legend(loc='lower left', fontsize=9, frameon=True)
    
    # Mark the zoom region on main plot with a dashed rectangle
    rect = Rectangle((0.9, 0), 0.2, 1.05, linewidth=1.5, edgecolor='black', 
                    facecolor='none', linestyle='--', alpha=0.5, zorder=1)
    ax_a.add_patch(rect)
    
    # --- PANEL B: Forensic Entropy (Task "How") ---
    ax_b = fig.add_subplot(gs[0, 1])
    
    print("   Computing forensic entropy...")
    n_range = np.arange(2, 201, 2)  # Cluster sizes from 2 to 200
    h_vals = []
    certainty_vals = []  # Attribution certainty = 1 - H/H_max
    
    for n in n_range:
        try:
            post = attribute_source(p_nsb, n, prior_type="flat").numpy()
            # Compute Shannon entropy
            h = -np.sum(post * np.log(post + 1e-10))
            h_vals.append(h)
            # Maximum entropy for uniform distribution over z in [1, n]
            h_max = np.log(n)
            certainty_vals.append(1 - h / (h_max + 1e-10))
        except:
            # If computation fails for large n, use previous value
            if len(h_vals) > 0:
                h_vals.append(h_vals[-1])
                certainty_vals.append(certainty_vals[-1])
            else:
                h_vals.append(0)
                certainty_vals.append(0)
    
    h_vals = np.array(h_vals)
    certainty_vals = np.array(certainty_vals)
    
    # Define Forensic Horizon (transition point)
    forensic_horizon = 50
    
    # Shade the two regimes
    # Forensic Window (n < 50): Active Attribution Zone
    ax_b.axvspan(0, forensic_horizon, color='green', alpha=0.2, zorder=0, 
                label="Forensic Window: High-Signal Seeding")
    # Information Oblivion (n > 50): Spectral Saturation
    ax_b.axvspan(forensic_horizon, n_range[-1], color='gray', alpha=0.2, zorder=0,
                label="Information Oblivion: Loss of Identifiability")
    
    # Primary axis: Entropy (use different color from panel (a))
    line1 = ax_b.plot(n_range, h_vals, color='#2E86AB', lw=3, label="Shannon Entropy $\\mathcal{H}(z|n)$", zorder=3)
    ax_b.set_xlabel("Cluster Size ($n$)", fontsize=12, fontweight='bold')
    ax_b.set_ylabel("Shannon Entropy of Founder Sizes $H(z|n)$", fontsize=12, fontweight='bold', color='#2E86AB')
    ax_b.tick_params(axis='y', labelcolor='#2E86AB')
    
    # Secondary axis: Attribution Certainty (use different color from panel (a))
    ax_b2 = ax_b.twinx()
    line2 = ax_b2.plot(n_range, certainty_vals, color='#FF6B35', lw=2.5, ls='--', 
                       alpha=0.8, label="Source Attribution Certainty", zorder=2)
    ax_b2.set_ylabel("Normalised Attribution Certainty", 
                     fontsize=12, fontweight='bold', color='#FF6B35')
    ax_b2.tick_params(axis='y', labelcolor='#FF6B35')
    ax_b2.set_ylim(0, 1.05)
    
    # Add "Forensic Horizon" vertical line
    ax_b.axvline(forensic_horizon, color='black', ls='--', lw=2, alpha=0.7, zorder=4)
    # Find certainty value at horizon for annotation
    horizon_idx = np.argmin(np.abs(n_range - forensic_horizon))
    horizon_certainty = certainty_vals[horizon_idx] if horizon_idx < len(certainty_vals) else 0
    ax_b.text(forensic_horizon, ax_b.get_ylim()[1] * 0.95, 'Forensic\nHorizon', 
             fontsize=10, ha='center', va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                      edgecolor='black', linewidth=1.5))
    
    # Combine legends (include shaded regions and Forensic Horizon)
    patch1 = Patch(facecolor='green', alpha=0.2, label="Forensic Window: High-Signal Seeding")
    patch2 = Patch(facecolor='gray', alpha=0.2, label="Information Oblivion: Loss of Identifiability")
    horizon_line = Line2D([0], [0], color='black', ls='--', lw=2.5, alpha=0.7, label="Forensic Horizon ($n = 50$): Maximum Resolution")
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # Add patches and horizon line to legend
    handles = [patch1, patch2, horizon_line] + lines
    legend_labels = [patch1.get_label(), patch2.get_label(), horizon_line.get_label()] + labels
    ax_b.legend(handles, legend_labels, loc='center right', fontsize=10, frameon=True)
    
    ax_b.set_title("(b) Forensic Information Horizon/Resolution", 
                   fontsize=14, fontweight='bold')
    ax_b.grid(alpha=0.3, linestyle='--')
    
    # --- PANEL C: Branching Volatility (Task "Next") ---
    ax_c = fig.add_subplot(gs[1, 0])
    
    print("   Computing branching volatility...")
    k_display = np.arange(min(20, k_max))  # Display up to k=19
    p_nsb_display = p_nsb[:len(k_display)].numpy()
    p_pois = get_poisson_p_dist(r0_learned, len(k_display)).numpy()
    p_nb = get_nb_p_dist(r0_learned, len(k_display), overdispersion=0.1).numpy()
    
    # Normalize for display
    p_nsb_display = p_nsb_display / p_nsb_display.sum()
    p_pois = p_pois / p_pois.sum()
    p_nb = p_nb / p_nb.sum()
    
    # Bar positions
    x_pos = np.arange(len(k_display))
    width = 0.25
    
    # Create grouped bars
    bars1 = ax_c.bar(x_pos - width, p_nsb_display, width, label="NSB (SARS/MERS)", 
                    color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3)
    bars2 = ax_c.bar(x_pos, p_pois, width, label="Poisson Baseline", 
                    color='#06A77D', alpha=0.6, edgecolor='black', linewidth=0.5, zorder=2)
    bars3 = ax_c.bar(x_pos + width, p_nb, width, label="NegBin Baseline", 
                    color='#A23B72', alpha=0.6, edgecolor='black', linewidth=0.5, zorder=2)
    
    # Highlight key regions: k=0 (extinction) and k > 5 (superspreading)
    ax_c.axvspan(-0.5, 0.5, alpha=0.15, color='red', zorder=0, label="Individual Extinction ($k=0$): Tight Loop around $G(1)=1$")
    ax_c.axvspan(5.5, len(k_display)-0.5, alpha=0.15, color='orange', zorder=0, 
                label="Superspreading Tail ($k > 5$): High-freq components near $G(1)=1$")
    
    ax_c.set_xlabel("Offspring Count ($k$)", fontsize=12, fontweight='bold')
    ax_c.set_ylabel("Offspring Probability $p_k$ (log scale)", fontsize=12, fontweight='bold')
    ax_c.set_title("(c) Superspreading Risk & Tail Probabilities", 
                   fontsize=14, fontweight='bold')
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([str(int(k)) for k in k_display])
    ax_c.set_yscale('log')
    ax_c.legend(loc='center right', fontsize=10, frameon=True)
    ax_c.grid(alpha=0.3, linestyle='--', axis='y')
    
    # --- PANEL D: Spectral Energy Dissipation ---
    ax_d = fig.add_subplot(gs[1, 1])
    
    print("   Computing spectral energy dissipation...")
    theta = np.linspace(0, 2 * np.pi, 1000)
    s_circle = np.exp(1j * theta)
    
    # Get baseline distributions (matching panel (a) colors)
    p_pois = get_poisson_p_dist(r0_learned, k_max)
    p_nb = get_nb_p_dist(r0_learned, k_max, overdispersion=0.1)
    
    # Compute PGFs for different generations
    generations = [1, 2, 3, 5, 10]
    colors_gen = plt.cm.plasma(np.linspace(0.9, 0.1, len(generations)))
    
    # Plot NSB generation PGFs
    for i, m in enumerate(generations):
        g_m = compute_generation_pgf(p_nsb, m, s_circle)
        ax_d.plot(g_m.real, g_m.imag, label=f"NSB Gen $m={m}$", 
                 color=colors_gen[i], lw=2.5, alpha=0.9, zorder=4-i)
    
    # Plot Poisson baseline for generation 1 (for comparison)
    g_pois_gen1 = compute_generation_pgf(p_pois, 1, s_circle)
    ax_d.plot(g_pois_gen1.real, g_pois_gen1.imag, color='#06A77D', ls='--', lw=2, 
             alpha=0.7, label="Poisson Baseline", zorder=2)
    
    # Plot NegBin baseline for generation 1 (for comparison)
    g_nb_gen1 = compute_generation_pgf(p_nb, 1, s_circle)
    ax_d.plot(g_nb_gen1.real, g_nb_gen1.imag, color='#A23B72', ls=':', lw=2, 
             alpha=0.7, label="NegBin Baseline", zorder=2)
    
    # Residual Shading: Area between Poisson and NSB (generation 1)
    # Find points where both are in the visible range
    g_nsb_gen1 = compute_generation_pgf(p_nsb, 1, s_circle)
    mask = (g_nsb_gen1.real >= 0.5) & (g_nsb_gen1.real <= 1.1) & (g_nsb_gen1.imag >= -0.3) & (g_nsb_gen1.imag <= 0.3)
    mask_pois = (g_pois_gen1.real >= 0.5) & (g_pois_gen1.real <= 1.1) & (g_pois_gen1.imag >= -0.3) & (g_pois_gen1.imag <= 0.3)
    
    # Interpolate to same length for filling
    if np.sum(mask) > 10 and np.sum(mask_pois) > 10:
        # Sample points for filling
        n_fill = min(200, len(g_nsb_gen1))
        indices = np.linspace(0, len(g_nsb_gen1)-1, n_fill, dtype=int)
        indices_pois = np.linspace(0, len(g_pois_gen1)-1, n_fill, dtype=int)
        
        g_nsb_fill = g_nsb_gen1[indices]
        g_pois_fill = g_pois_gen1[indices_pois]
        
        # Create polygon for shading
        fill_x = np.concatenate([g_nsb_fill.real, g_pois_fill.real[::-1]])
        fill_y = np.concatenate([g_nsb_fill.imag, g_pois_fill.imag[::-1]])
        ax_d.fill(fill_x, fill_y, color='orange', alpha=0.15, zorder=1, 
                 label="Overdispersion Gap")
    
    # Unit circle backdrop (faint)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='lightgray', 
                            linestyle='--', linewidth=1, alpha=0.2, zorder=0)
    ax_d.add_patch(unit_circle)
    
    # Vertical line at x=1
    ax_d.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Normalization anchor (matching exp_task_who.py style)
    ax_d.scatter([1], [0], color='red', s=100, zorder=6, marker='o', 
                edgecolors='black', linewidths=2, label="$G(1)=1 \\Leftrightarrow \\sum p_k = 1$")
    
    # Conformal zoom into critical region
    ax_d.set_xlim(0.5, 1.1)
    ax_d.set_ylim(-0.3, 0.3)
    ax_d.set_aspect('equal', adjustable='box')
    
    ax_d.set_xlabel("Re($G_m(s)$)", fontsize=12, fontweight='bold')
    ax_d.set_ylabel("Im($G_m(s)$)", fontsize=12, fontweight='bold')
    ax_d.set_title("(d) Generational Loss of Forensic Memory", 
                   fontsize=14, fontweight='bold')
    ax_d.legend(loc='upper left', fontsize=9, frameon=True)
    ax_d.grid(alpha=0.2, linestyle='--')
    
    # Add colorbar for generation mapping (use smaller fraction to not affect panel size)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                               norm=plt.Normalize(vmin=1, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_d, fraction=0.035, pad=0.02)
    cbar.set_label("Generation ($m$): $G_m(s) = G(G_{m-1}(s))$", fontsize=9, rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=8)
    
    # Title removed to save space
    
    save_figure(fig, "exp_task_how_next")
    print(f"Dashboard saved to '{CONFIG['output_dir_figures'] / 'exp_task_how_next.pdf'}'")
    plt.close()

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Forensic Forecasting Dashboard: Task 'How' & 'Next'")
    print("=" * 70)
    
    # Load and split data
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
    p_learned = model.predict_pmf(k_max=150)
    p_learned = torch.from_numpy(p_learned).float()
    p_learned = p_learned / p_learned.sum()
    
    metrics = compute_how_metrics(p_learned)
    print(f"   Learned R0: {metrics['r0']:.4f}")
    print(f"   Extinction Probability: {metrics['extinction_prob']:.4f}")
    print(f"   Distribution support: k_max = {len(p_learned)}")
    
    # Generate dashboard
    print(f"\n4. Generating Forensic Forecasting Dashboard...")
    plot_forecasting_dashboard(p_learned, pathogen_name="SARS/MERS")
    
    print("\n" + "=" * 70)
    print("Dashboard generation complete!")
    print("=" * 70)

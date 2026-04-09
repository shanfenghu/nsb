from __future__ import annotations

"""
Shared utilities for the `experiments/rebuttal/` scripts.

Design goals:
- Keep experiment scripts short and readable.
- Provide consistent metrics (TV/KL, posteriors, timing).
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from nsb.model import NSB
from nsb.spectral_engine import SpectralEngine
from nsb.task_who import get_prior


@dataclass(frozen=True)
class Timing:
    seconds: float

    @property
    def ms(self) -> float:
        return self.seconds * 1e3


def set_global_seed(seed: int) -> None:
    """
    Set RNG seeds for numpy and torch (CPU).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total variation distance: TV(p,q) = 0.5 * ||p - q||_1.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL divergence KL(p||q) with small epsilon clipping for numerical stability.

    Returns value in nats (log base e).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def load_outbreaktrees_counts(data_path: Path) -> np.ndarray:
    """
    Load outbreaktrees offspring counts from a CSV file.
    """
    df = pd.read_csv(data_path)
    if "offspring_count" not in df.columns:
        raise ValueError(f"Expected column 'offspring_count' in {data_path}")
    counts = df["offspring_count"].to_numpy()
    return counts.astype(int)


def train_nsb_on_outbreaktrees(
    *,
    data_path: Path,
    k_max: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
    device: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """
    Trains NSB on outbreaktrees counts and returns a normalized PMF (length k_max+1).
    """
    set_global_seed(seed)
    counts = load_outbreaktrees_counts(data_path)

    model = NSB(hidden_dim=hidden_dim)
    if device is not None:
        model.device = torch.device(device)
        model.cell.to(model.device)
        if model.h0 is not None:
            model.h0 = model.h0.to(model.device)

    model.fit(counts, epochs=epochs, lr=lr, batch_size=batch_size)
    pmf = model.predict_pmf(k_max=k_max)

    pmf = np.asarray(pmf, dtype=float)
    pmf_sum = float(pmf.sum())
    if pmf_sum <= 0:
        raise RuntimeError("Learned PMF had non-positive mass.")
    pmf_norm = pmf / pmf_sum
    k = np.arange(len(pmf_norm), dtype=float)
    r0 = float(np.sum(k * pmf_norm))
    meta = {
        "pmf_sum_before_norm": pmf_sum,
        "truncation_gap_before_norm": 1.0 - pmf_sum,
        "r0_after_norm": r0,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "k_max": k_max,
    }
    return pmf_norm, meta


def maybe_load_cached_pmf(cache_path: Optional[Path]) -> Optional[np.ndarray]:
    """
    Load a cached PMF numpy array if present, otherwise return None.
    """
    if cache_path is None:
        return None
    if not cache_path.exists():
        return None
    arr = np.load(cache_path)
    return np.asarray(arr, dtype=float)


def save_cached_pmf(cache_path: Path, pmf: np.ndarray) -> None:
    """
    Save a PMF numpy array to `cache_path`, creating parent dirs as needed.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.asarray(pmf, dtype=float))


def posterior_z_given_c(
    p_dist: np.ndarray | torch.Tensor,
    *,
    n: int,
    prior_type: str = "flat",
    prior_params: Optional[dict] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Returns posterior over z=1..n as a numpy vector of length n.
    """
    if isinstance(p_dist, np.ndarray):
        p = torch.from_numpy(p_dist).float()
    else:
        p = p_dist.float()

    if device is not None:
        p = p.to(device)

    likelihoods = SpectralEngine.compute_likelihood_surface(p, n)
    prior = get_prior(n, prior_type=prior_type, params=prior_params, device=str(likelihoods.device))
    posterior = likelihoods * prior
    denom = posterior.sum()
    if denom <= 0:
        posterior = torch.ones(n, device=likelihoods.device) / n
    else:
        posterior = posterior / denom
    return posterior.detach().cpu().numpy()


def optimal_fft_length(n: int, K: int) -> int:
    """
    Small helper for choosing a power-of-two FFT length for n-fold convolution support.
    """
    max_length = n * (K - 1) + 1
    if max_length <= 1:
        return 1
    return 2 ** int(math.ceil(math.log2(max_length)))


def piecewise_likelihood_surface(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    n1: int,
    n2: int,
    n: int,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Computes likelihoods for z=1..n under piecewise i.n.i.d. offspring laws:
      P(C=n | Z=z) = (z/n) * [s^{n-z}] G1(s)^{n1} G2(s)^{n2}.
    """
    if n1 + n2 != n:
        raise ValueError("Require n1 + n2 == n")

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if len(p1) != len(p2):
        raise ValueError("Require p1 and p2 to have same length (same truncation K).")

    K = len(p1)
    L = optimal_fft_length(n=n, K=K)

    dev = torch.device(device) if device is not None else torch.device("cpu")
    p1_t = torch.zeros(L, device=dev, dtype=torch.complex64)
    p2_t = torch.zeros(L, device=dev, dtype=torch.complex64)
    p1_t[:K] = torch.from_numpy(p1).to(dev).to(torch.complex64)
    p2_t[:K] = torch.from_numpy(p2).to(dev).to(torch.complex64)

    p1_hat = torch.fft.fft(p1_t)
    p2_hat = torch.fft.fft(p2_t)
    q_hat = torch.pow(p1_hat, n1) * torch.pow(p2_hat, n2)
    q = torch.fft.ifft(q_hat)

    z = torch.arange(1, n + 1, device=dev, dtype=torch.float32)
    coeffs = q[n - z.long()].real
    lik = (z / n) * coeffs
    lik = torch.clamp(lik, min=0.0)
    return lik.detach().cpu().numpy()


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, Timing(t1 - t0)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

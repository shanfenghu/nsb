"""
NSB Task Module: "How" (Structural Fingerprinting)

This module addresses the forensic question "How certain can we be about source
attribution?" by computing structural metrics of the learned offspring distribution:

1. Shannon Entropy: Measures uncertainty in the offspring distribution
2. Basic Reproductive Number (R0): Mean offspring count R0 = E[k] = sum(k * p_k)
3. Extinction Probability (q): Fixed point of PGF, q = G(q)

The extinction probability is computed using Newton-Raphson root finding for
supercritical processes (R0 > 1), with q = 1 for subcritical processes (R0 ≤ 1).
"""

import torch
from typing import Dict


def compute_how_metrics(p_dist: torch.Tensor) -> Dict[str, float]:
    """
    Computes structural metrics of the offspring distribution for forensic analysis.

    This function calculates three key metrics:
    - Entropy: Shannon entropy H = -sum(p_k * log(p_k)) measuring distribution uncertainty
    - R0: Basic reproductive number R0 = sum(k * p_k) measuring mean transmission
    - Extinction Probability: Fixed point q = G(q) measuring outbreak survival probability

    Args:
        p_dist (torch.Tensor): Offspring probability distribution p_k (will be normalized)

    Returns:
        Dict[str, float]: Dictionary with keys:
            - "entropy": Shannon entropy (bits)
            - "r0": Basic reproductive number
            - "extinction_prob": Extinction probability q ∈ [0, 1]
    """
    p_dist = p_dist / p_dist.sum()
    K = len(p_dist)
    k_indices = torch.arange(K, device=p_dist.device).float()

    # 1. Offspring Entropy (H)
    eps = 1e-12
    entropy = -torch.sum(p_dist * torch.log(p_dist + eps)).item()

    # 2. Basic Reproductive Number (R0)
    r0 = torch.sum(k_indices * p_dist).item()

    # 3. Extinction Probability (q) via Newton's Method
    extinction_prob = _solve_extinction_newton(p_dist, r0)

    return {
        "entropy": float(entropy),
        "r0": float(r0),
        "extinction_prob": float(extinction_prob)
    }


def _solve_extinction_newton(p_dist: torch.Tensor, r0: float) -> float:
    """
    Solves the fixed-point equation q = G(q) using Newton-Raphson method.

    The extinction probability q is the smallest non-negative solution to q = G(q),
    where G(s) = sum_k p_k * s^k is the probability generating function. For
    subcritical processes (R0 ≤ 1), q = 1. For supercritical processes (R0 > 1),
    we solve f(s) = G(s) - s = 0 using Newton-Raphson with initial guess s = 0.5.

    Args:
        p_dist (torch.Tensor): Offspring probability distribution (normalized)
        r0 (float): Basic reproductive number (for early termination if R0 ≤ 1)

    Returns:
        float: Extinction probability q ∈ [0, 1]
    """
    if r0 <= 1.0:
        return 1.0

    # For supercritical processes (R0 > 1), we start at s=0.5.
    # This ensures we find the smallest non-negative root (q).
    s = torch.tensor(0.5, device=p_dist.device)
    k_indices = torch.arange(len(p_dist), device=p_dist.device).float()
    
    # Derivative indices: k * s^(k-1)
    k_deriv = torch.arange(len(p_dist), device=p_dist.device).float()

    for _ in range(20): # Newton usually converges in < 10 steps
        # G(s) = sum(p_k * s^k)
        g_s = torch.sum(p_dist * torch.pow(s, k_indices))
        
        # G'(s) = sum(k * p_k * s^(k-1))
        # Note: at s=0, s^(k-1) is only defined for k=1 (where it is 1).
        # We use torch.pow and handle the 0^0 case implicitly.
        g_prime_s = torch.sum(k_deriv * p_dist * torch.pow(s, k_deriv - 1).nan_to_num(0.0))
        
        f_s = g_s - s
        f_prime_s = g_prime_s - 1.0
        
        # Avoid division by zero (unlikely in supercritical regime)
        if torch.abs(f_prime_s) < 1e-10:
            break
            
        s_next = s - f_s / f_prime_s
        
        # Clamp for stability in [0, 1]
        s_next = torch.clamp(s_next, 0.0, 1.0)
        
        if torch.abs(s_next - s) < 1e-8:
            s = s_next
            break
        s = s_next
            
    return s.item()
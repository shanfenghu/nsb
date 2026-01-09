"""
NSB Task Module: "Who" (Source Attribution)

This module implements Algorithm 3: One-Pass Patient Zero Attribution, which solves
the forensic question "Who started the outbreak?" by inferring the number of founders
(patient zeros) Z from an observed cluster size C = n.

The algorithm leverages:
1. The Neural Otter-Dwass identity: P(C=n | Z=z) = (z/n) * [s^{n-z}] G(s)^n
2. Spectral methods (FFT) for efficient n-fold convolution
3. Bayesian inference with epidemiological priors (Flat, Poisson, Negative Binomial)

The method achieves O(nK log(nK)) complexity, enabling real-time source attribution
even for large clusters.
"""

import torch
import numpy as np
from typing import Optional, Dict, Union


def get_prior(n: int, prior_type: str = "flat", params: Optional[Dict] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generates the epidemiological prior π(z) for the number of founders (patient zeros).

    Three prior archetypes are supported, each representing different epidemiological
    scenarios:
    - "flat": Clinical/high-traffic settings with uniform risk of seeding
    - "community": Sparse, independent introductions (Poisson distribution)
    - "clustered": Overdispersed seeding events (Negative Binomial distribution)

    Args:
        n: Maximum number of founders to consider (support is z ∈ {1, 2, ..., n})
        prior_type: Type of prior ("flat", "community", or "clustered")
        params: Optional dictionary of prior parameters:
            - For "community": {"lambda": float} (Poisson mean)
            - For "clustered": {"r": float, "p": float} (NB shape and success prob)
        device: Device to create tensors on ("cpu" or "cuda")

    Returns:
        torch.Tensor: Normalized prior distribution π(z) for z ∈ {1, 2, ..., n}
    """
    z_range = torch.arange(1, n + 1, device=device).float()
    
    if prior_type == "flat":
        # Archetype: Clinical and High-Traffic Settings (Constant/Sustained)
        return torch.ones(n, device=device) / n
        
    elif prior_type == "community":
        # Archetype: Community Introduction (Sparse/Independent)
        # Using Poisson: pi(z) = (lambda^z * e^-lambda) / z!
        lam = params.get("lambda", 2.0) if params else 2.0
        # Compute in log-space for numerical stability
        log_prior = z_range * torch.log(torch.tensor(lam)) - lam - torch.lgamma(z_range + 1)
        prior = torch.exp(log_prior)
        return prior / prior.sum()
        
    elif prior_type == "clustered":
        # Archetype: Clustered Seeding (Overdispersed/Super-Seeding)
        # Using Negative Binomial: Captures high variance in introductions
        r = params.get("r", 2.0) if params else 2.0 # Number of successes
        p = params.get("p", 0.5) if params else 0.5 # Probability of success
        
        # NB PMF: exp(log_gamma(z+r) - log_gamma(z+1) - log_gamma(r) + r*log(1-p) + z*log(p))
        log_prior = (torch.lgamma(z_range + r) - torch.lgamma(z_range + 1) - torch.lgamma(torch.tensor(r)) +
                     r * torch.log(torch.tensor(1 - p)) + z_range * torch.log(torch.tensor(p)))
        prior = torch.exp(log_prior)
        return prior / prior.sum()
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}. Choose from 'flat', 'community', or 'clustered'.")


def attribute_source(
    p_dist: torch.Tensor, 
    n: int, 
    prior_type: str = "flat", 
    prior_params: Optional[Dict] = None
) -> torch.Tensor:
    """
    Implementation of Algorithm 3: One-Pass Patient Zero Attribution.

    This function computes the posterior distribution P(Z=z | C=n) using the
    Neural Otter-Dwass identity and spectral methods. The algorithm:
    1. Pads the offspring distribution to length L = 2^ceil(log2(n*(K-1)+1))
    2. Computes G(s)^n via FFT: p_hat^n
    3. Inverts via IFFT to get coefficients of G(s)^n
    4. Extracts likelihoods using Otter-Dwass: P(C=n|Z=z) = (z/n) * [s^{n-z}] G(s)^n
    5. Applies Bayesian update: P(Z=z|C=n) ∝ P(C=n|Z=z) * π(z)
    
    Args:
        p_dist (torch.Tensor): Offspring distribution {p_k} of length K.
                              Must be normalized (sums to ~1.0).
        n (int): Observed aggregate cluster size (must be positive).
        prior_type (str): Prior type: 'flat', 'community', or 'clustered'.
        prior_params (dict, optional): Parameters for the chosen prior:
            - For "community": {"lambda": float}
            - For "clustered": {"r": float, "p": float}
        
    Returns:
        torch.Tensor: The posterior distribution P(Z=z | C=n) for z ∈ {1, 2, ..., n}.
                      Normalized to sum to 1.0.
    """
    device = p_dist.device
    K = len(p_dist)

    # 1. Initialize: Forensic Shield (L >= n(K-1)+1)
    max_len = n * (K - 1) + 1
    L = 2 ** int(np.ceil(np.log2(max_len)))

    # 2. Pad to transform length L
    p_padded = torch.zeros(L, device=device, dtype=torch.complex64)
    p_padded[:K] = p_dist.to(torch.complex64)

    # 3. Spectral Mapping (FFT)
    p_hat = torch.fft.fft(p_padded)

    # 4. Spectral Power (n-fold convolution dual)
    q_hat = torch.pow(p_hat, n)

    # 5. Inverse Transformation (IFFT)
    q = torch.fft.ifft(q_hat)

    # 6. Likelihood Extraction & Bayesian Update
    # Extract z/n * Re(q_{n-z})
    z_indices = torch.arange(1, n + 1, device=device)
    # n-z indices: for z=1 (index n-1), for z=n (index 0)
    q_coeffs = q[n - z_indices.long()].real
    likelihoods = (z_indices.float() / n) * q_coeffs
    
    # Ensure physical validity (non-negative)
    likelihoods = torch.clamp(likelihoods, min=0.0)

    # 7. Apply Prior
    pi_z = get_prior(n, prior_type, prior_params, device=device)
    posterior = likelihoods * pi_z

    # 8. Normalize and return
    denom = posterior.sum()
    if denom == 0:
        # Fallback in case of total numerical vanishing (unlikely with Shield)
        return torch.ones(n, device=device) / n
        
    return posterior / denom
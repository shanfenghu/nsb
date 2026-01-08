"""
NSB Task Module: "Who" (Source Attribution)
Implements Algorithm 3: One-Pass Patient Zero Attribution using the 
Neural Otter-Dwass identity and epidemiological priors.
"""

import torch
import numpy as np
from typing import Optional, Dict, Union


def get_prior(n: int, prior_type: str = "flat", params: Optional[Dict] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generates the epidemiological prior pi(z) for the number of founders.
    Realigned with the three archetypes: Poisson, Flat, and Negative Binomial.
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
    
    Args:
        p_dist (torch.Tensor): Offspring distribution {p_k} of length K.
        n (int): Observed aggregate cluster size.
        prior_type (str): 'flat', 'community', or 'clustered'.
        prior_params (dict): Parameters for the chosen prior.
        
    Returns:
        torch.Tensor: The posterior distribution P(Z=z | C=n) for z in {1..n}.
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
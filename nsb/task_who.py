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
    """
    z_range = torch.arange(1, n + 1, device=device).float()
    
    if prior_type == "flat":
        # Uninformative: every founder count is equally likely
        return torch.ones(n, device=device) / n
        
    elif prior_type == "sparse":
        # Epidemic Sparse: Exponential decay representing rare introduction events
        lam = params.get("lambda", 0.5) if params else 0.5
        prior = torch.exp(-lam * z_range)
        return prior / prior.sum()
        
    elif prior_type == "informed":
        # Surveillance-Informed: Gaussian centered on suspected entry count
        mu = params.get("mu", 1.0) if params else 1.0
        sigma = params.get("sigma", 1.0) if params else 1.0
        prior = torch.exp(-0.5 * ((z_range - mu) / sigma) ** 2)
        return prior / prior.sum()
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


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
        prior_type (str): 'flat', 'sparse', or 'informed'.
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
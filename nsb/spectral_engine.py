"""
Implementation of the Spectral Inversion Engine for the NSB framework.

This module provides the mathematical backbone for forensic inference, 
leveraging the duality between probability generating functions (PGFs) 
and the complex unit circle to solve combinatorial branching problems 
analytically rather than stochastically.
"""

import torch
import numpy as np
from typing import Optional


class SpectralEngine:
    """
    The Spectral Inversion Engine for the Neural Stick-Breaking process.

    This engine implements the Analytic Inversion Bridge, utilizing the 
    Fast Fourier Transform (FFT) and the Otter-Dwass identity to extract 
    exact branching likelihoods from learned offspring distributions.
    """

    @staticmethod
    def _get_optimal_fft_length(n: int, K: int) -> int:
        """
        Calculates the zero-padding length L to prevent spectral aliasing.
        
        For n-fold convolution of a length-K distribution, the result has length
        at most n*(K-1) + 1. We pad to the next power of 2 to ensure we capture
        all coefficients without aliasing.
        
        Args:
            n: Cluster size (number of convolutions)
            K: Length of the distribution
            
        Returns:
            L: Padding length (power of 2) that prevents aliasing
        """
        # Maximum length after n-fold convolution: n*(K-1) + 1
        max_length = n * (K - 1) + 1
        # Round up to next power of 2 for FFT efficiency
        return 2 ** int(np.ceil(np.log2(max_length)))

    @classmethod
    def compute_likelihood_surface(
        cls, 
        p_dist: torch.Tensor, 
        n: int, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Computes the full likelihood surface P(C=n | Z=z) for all z in {1..n}.

        This implements the One-Pass Attribution algorithm (Algorithm 2) using the
        Neural Otter-Dwass identity (Lemma 1). The algorithm leverages the duality
        between n-fold convolution and complex exponentiation in the spectral domain:

        1. Zero-padding to length L = 2^ceil(log2(n*(K-1)+1)) prevents aliasing
        2. Forward FFT maps p_dist to spectral domain: p_hat = FFT(p_padded)
        3. Spectral powering: q_hat = p_hat^n (equivalent to n-fold convolution)
        4. Inverse FFT maps back: q = IFFT(q_hat)
        5. Extract likelihoods via Otter-Dwass: P(C=n|Z=z) = (z/n) * Re(q[n-z])

        Args:
            p_dist (torch.Tensor): The learned offspring distribution {p_k}.
                                   Shape: (K,), must be normalized
            n (int): The observed aggregate cluster size (must be positive)
            device (torch.device, optional): Device to perform computations on.
                                            Defaults to p_dist.device

        Returns:
            torch.Tensor: Likelihood vector L where L[z-1] = P(C=n | Z=z) for z ∈ {1..n}.
                          Shape: (n,), all values are non-negative
        """
        # --- Validation Checks ---
        if n <= 0:
            raise ValueError(f"Cluster size n must be positive, got {n}.")
        
        if not torch.is_tensor(p_dist):
            p_dist = torch.tensor(p_dist, dtype=torch.float32)

        # Ensure the distribution is valid (sums to ~1.0)
        # Note: We allow slight deviation due to truncation in early training
        p_sum = p_dist.sum().item()
        if p_sum > 1.001:
             raise ValueError(f"p_dist is not a valid distribution (sum={p_sum:.4f})")

        if device is None:
            device = p_dist.device

        # --- Phase 1: Zero-Padding (The Forensic Shield) ---
        # Padding to L ensures we resolve all n-fold frequencies without aliasing.
        K = len(p_dist)
        L = cls._get_optimal_fft_length(n, K)
        
        # Pad with zeros to length L
        p_padded = torch.zeros(L, device=device, dtype=torch.complex64)
        p_padded[:K] = p_dist[:K].to(torch.complex64)

        # --- Phase 2: Spectral Mapping (The Bridge) ---
        # Map p_dist to the complex unit circle (Forward FFT)
        p_hat = torch.fft.fft(p_padded)

        # --- Phase 3: Spectral Powering (Convolution Dual) ---
        # Complex exponentiation is the spectral dual of n-fold self-convolution.
        # This implicitly sums over all 4^n possible transmission topologies.
        q_hat = torch.pow(p_hat, n)

        # --- Phase 4: Analytic Inversion (IFFT) ---
        # Map back to coefficient space to get coefficients of G(s)^n
        q = torch.fft.ifft(q_hat)

        # --- Phase 5: Combinatorial Extraction (Otter-Dwass) ---
        # We extract z/n * Re(q_{n-z}) for each potential founder count z.
        # This resolves the likelihood surface in a single log-linear pass.
        likelihoods = torch.zeros(n, device=device)
        
        # z ranges from 1 to n
        z_indices = torch.arange(1, n + 1, device=device, dtype=torch.float32)
        
        # Coefficients are at q[n-z]. For z=1, coeff is q[n-1]. For z=n, coeff is q[0].
        # We use .real to discard numerical imaginary artifacts (Hermitian Check).
        q_coeffs = q[n - z_indices.long()].real
        
        likelihoods = (z_indices / n) * q_coeffs

        # Final physical validity check: likelihoods must be non-negative
        return torch.clamp(likelihoods, min=0.0)

    @classmethod
    def compute_spectral_radius(cls, weights: torch.Tensor) -> float:
        """
        Calculates the spectral radius ρ(W_h) of the recurrent weight matrix.

        The spectral radius is the magnitude of the largest eigenvalue, which
        determines the stability and tail decay properties of the NSB process
        (Theorem 6). For ρ < 1, the process is stable with exponential tail decay.
        For ρ = 1, the process is at criticality with heavy tails.

        Args:
            weights (torch.Tensor): Square weight matrix (e.g., W_h from NSB cell)
                                    Shape: (d, d) where d is the hidden dimension

        Returns:
            float: Spectral radius ρ = max(|λ_i|) where λ_i are eigenvalues
        """
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("Weights must be a square matrix.")
            
        # Calculate eigenvalues
        eigenvalues = torch.linalg.eigvals(weights)
        # Return the magnitude of the dominant eigenvalue
        return float(torch.max(torch.abs(eigenvalues)).item())

    @classmethod
    def check_hermitian_symmetry(cls, spectral_vec: torch.Tensor) -> bool:
        """
        Verifies if the spectral vector maintains Hermitian symmetry.

        For a real-valued probability distribution, the FFT output must satisfy
        Hermitian symmetry: p_hat[m] = conj(p_hat[L-m]) for m = 1, ..., L//2.
        This ensures that the inverse FFT produces purely real coefficients,
        which is required for valid probability distributions.

        Args:
            spectral_vec (torch.Tensor): Complex spectral vector from FFT
                                         Shape: (L,)

        Returns:
            bool: True if Hermitian symmetry is maintained (within tolerance),
                  False otherwise
        """
        # Complex conjugate symmetry: hat{p}_m = conj(hat{p}_{L-m})
        L = len(spectral_vec)
        for m in range(1, L // 2):
            if not torch.allclose(spectral_vec[m], spectral_vec[L - m].conj(), atol=1e-5):
                return False
        return True
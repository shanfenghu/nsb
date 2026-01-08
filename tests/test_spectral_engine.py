"""
Unit tests for the Spectral Inversion Engine (nsb/spectral_engine.py).
Focuses on generalized robustness, mathematical identities, and edge cases
without relying on specific tutorial parameters.
"""

import pytest
import torch
import numpy as np
from nsb.spectral_engine import SpectralEngine

class TestSpectralEngine:
    """
    Robust test suite for the SpectralEngine using generalized mathematical 
    cases and reference comparisons.
    """

    # --- 1. Combinatorial Identity Tests ---

    def test_dirac_delta_at_zero(self):
        """
        Case: p_0 = 1 (No one can ever produce offspring).
        Identity: To see a cluster of size n, there must have been exactly n founders.
        P(C=n | Z=z) should be 1.0 if z=n, and 0.0 otherwise.
        """
        p = torch.tensor([1.0, 0.0, 0.0, 0.0])
        n = 10
        likelihoods = SpectralEngine.compute_likelihood_surface(p, n)
        
        # Likelihood should be 0 for z=1..n-1 and 1.0 for z=n
        expected = torch.zeros(n)
        expected[-1] = 1.0
        assert torch.allclose(likelihoods, expected, atol=1e-6)

    def test_n1_identity_random(self):
        """
        Case: Cluster size n=1.
        Identity: P(C=1 | Z=z) is only possible if z=1 (the founder itself).
        Likelihood should be 1/1 * coeff(G^1)_{1-1=0} = p_0.
        """
        # Generate a random valid distribution
        p = torch.rand(10)
        p /= p.sum()
        
        likelihoods = SpectralEngine.compute_likelihood_surface(p, n=1)
        assert torch.allclose(likelihoods[0], p[0], atol=1e-6)

    # --- 2. Reference Comparison (Numerical Validation) ---

    def test_against_numpy_convolution(self):
        """
        Validates the Spectral Bridge against time-domain O(n^2) convolution.
        For small n, NumPy's convolve is the ground truth for p*n.
        """
        p_np = np.array([0.4, 0.3, 0.2, 0.1])
        n = 5
        
        # Compute n-fold self-convolution p*n manually in time domain
        res_np = p_np
        for _ in range(n - 1):
            res_np = np.convolve(res_np, p_np)
        
        # Spectral Engine Calculation
        p_torch = torch.from_numpy(p_np).float()
        likelihoods = SpectralEngine.compute_likelihood_surface(p_torch, n)
        
        # Verify Otter-Dwass: L_z = z/n * res_np[n-z]
        for z in range(1, n + 1):
            expected_val = (z / n) * res_np[n - z]
            assert torch.allclose(likelihoods[z-1], torch.tensor(expected_val).float(), atol=1e-5)

    # --- 3. Robustness and Scaling ---

    @pytest.mark.parametrize("n_size", [16, 128, 512, 1024])
    def test_high_n_stability(self, n_size):
        """
        Tests the engine at various scales to ensure the log-linear 
        FFT logic and power-of-2 padding remain numerically stable.
        """
        p = torch.zeros(20)
        p[0] = 0.5
        p[1] = 0.5 # Simple branching
        
        try:
            likelihoods = SpectralEngine.compute_likelihood_surface(p, n_size)
            assert likelihoods.shape == (n_size,)
            assert torch.all(likelihoods >= 0.0)
            assert not torch.any(torch.isnan(likelihoods))
        except Exception as e:
            pytest.fail(f"SpectralEngine failed at n={n_size} with error: {e}")

    def test_device_agnosticism(self):
        """Ensures the engine respects the device of the input tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        p_cuda = torch.tensor([0.5, 0.5]).cuda()
        n = 10
        likelihoods = SpectralEngine.compute_likelihood_surface(p_cuda, n)
        assert likelihoods.is_cuda

    # --- 4. Matrix Dynamics and Stability ---

    def test_spectral_radius_generalized(self):
        """
        Tests the spectral radius utility with various matrix structures 
        (Identity, Zero, Nilpotent, and Random).
        """
        # Identity
        assert SpectralEngine.compute_spectral_radius(torch.eye(5)) == 1.0
        
        # Zero
        assert SpectralEngine.compute_spectral_radius(torch.zeros(3, 3)) == 0.0
        
        # Nilpotent (all eigenvalues are 0)
        W_nil = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        assert SpectralEngine.compute_spectral_radius(W_nil) == 0.0
        
        # Symmetric (real eigenvalues)
        W_sym = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        # Eigenvalues are 3 and 1
        assert torch.allclose(torch.tensor(SpectralEngine.compute_spectral_radius(W_sym)), torch.tensor(3.0))

    # --- 5. Error Handling and Constraints ---

    def test_invalid_distribution_sum(self):
        """Verifies that the engine catches non-normalized input distributions."""
        p_invalid = torch.tensor([2.0, 2.0])
        with pytest.raises(ValueError, match="not a valid distribution"):
            SpectralEngine.compute_likelihood_surface(p_invalid, 5)

    def test_hermitian_check_random(self):
        """
        Generates random real signals to verify the Hermitian symmetry 
        check works for non-tutorial data.
        """
        p = torch.rand(32)
        p_hat = torch.fft.fft(p.to(torch.complex64))
        assert SpectralEngine.check_hermitian_symmetry(p_hat)
        
        # Intentionally break symmetry
        p_hat[5] += 10.0j
        assert not SpectralEngine.check_hermitian_symmetry(p_hat)
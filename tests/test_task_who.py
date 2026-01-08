"""
Unit tests for the Source Attribution task (task_who.py).
Validates Bayesian posterior calculation, prior influence, and numerical 
consistency across diverse outbreak scenarios.
"""

import pytest
import torch
import numpy as np
from nsb.task_who import attribute_source, get_prior

class TestTaskWho:
    """
    Test suite for Patient Zero Attribution logic and Bayesian priors.
    """

    @pytest.fixture
    def mock_dist(self):
        """A simple, valid offspring distribution (K=3)."""
        return torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

    # --- 1. Prior Generation Tests ---

    def test_get_prior_flat(self):
        """Verifies the uniform prior is truly flat and normalized."""
        n = 10
        prior = get_prior(n, prior_type="flat")
        assert torch.allclose(prior, torch.ones(n) / n)
        assert torch.allclose(prior.sum(), torch.tensor(1.0))

    def test_get_prior_sparse(self):
        """Verifies that the sparse prior exhibits exponential decay."""
        n = 5
        # lambda=1.0 means pi(z) ~ exp(-z)
        prior = get_prior(n, prior_type="sparse", params={"lambda": 1.0})
        # pi(1) should be significantly greater than pi(5)
        assert prior[0] > prior[4]
        assert torch.allclose(prior.sum(), torch.tensor(1.0))

    def test_get_prior_informed(self):
        """Verifies the informed prior is centered on the specified mean."""
        n = 10
        mu = 5.0
        prior = get_prior(n, prior_type="informed", params={"mu": mu, "sigma": 1.0})
        # Peak should be at index 4 (z=5)
        assert torch.argmax(prior).item() == 4 
        assert torch.allclose(prior.sum(), torch.tensor(1.0))

    # --- 2. Integrated Attribution Tests ---

    def test_attribution_normalization(self, mock_dist):
        """Ensures the posterior always sums to 1.0 regardless of prior type."""
        n = 20
        priors = ["flat", "sparse", "informed"]
        for p_type in priors:
            post = attribute_source(mock_dist, n, prior_type=p_type)
            assert torch.allclose(post.sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.all(post >= 0.0)

    def test_bayesian_mode_shift(self):
        """
        Verify that the sparse prior shifts the MAP estimate toward a 
        lower number of founders compared to the flat prior.
        """
        # Dist where high founder count is slightly likely under flat prior
        p_dist = torch.tensor([0.2, 0.4, 0.4]) 
        n = 10
        
        post_flat = attribute_source(p_dist, n, prior_type="flat")
        post_sparse = attribute_source(p_dist, n, prior_type="sparse", prior_params={"lambda": 2.0})
        
        map_flat = torch.argmax(post_flat).item()
        map_sparse = torch.argmax(post_sparse).item()
        
        # Sparse prior should pull the peak toward z=1
        assert map_sparse <= map_flat

    def test_informed_prior_dominance(self, mock_dist):
        """
        If we have a very strong (low sigma) informed prior, the posterior 
        mode should match the prior mean regardless of the cluster size.
        """
        n = 50
        target_mu = 3.0
        # Extremely tight sigma (near Delta distribution)
        params = {"mu": target_mu, "sigma": 0.01}
        
        post = attribute_source(mock_dist, n, prior_type="informed", prior_params=params)
        
        # Mode should be exactly z=3 (index 2)
        assert torch.argmax(post).item() == 2

    # --- 3. Robustness and Edge Cases ---

    def test_single_node_outbreak(self, mock_dist):
        """For n=1, the only possible founder count is z=1."""
        n = 1
        post = attribute_source(mock_dist, n)
        assert len(post) == 1
        assert post[0].item() == 1.0

    @pytest.mark.parametrize("n_val", [10, 50, 100])
    def test_scaling_consistency(self, n_val):
        """Ensures logic holds across small and medium scales."""
        p_dist = torch.tensor([0.4, 0.3, 0.2, 0.1])
        post = attribute_source(p_dist, n_val, prior_type="flat")
        assert post.shape == (n_val,)

    def test_device_compatibility(self, mock_dist):
        """Ensures tensors stay on the correct device if CUDA is used."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        p_cuda = mock_dist.cuda()
        post = attribute_source(p_cuda, n=10)
        assert post.is_cuda
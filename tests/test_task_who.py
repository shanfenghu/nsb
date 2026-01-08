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

    def test_get_prior_community(self):
        """Verifies that the community prior (Poisson) exhibits appropriate decay."""
        n = 5
        # lambda=1.0 means Poisson with mean 1.0
        prior = get_prior(n, prior_type="community", params={"lambda": 1.0})
        # For Poisson with lambda=1.0, pi(1) should be greater than pi(5)
        assert prior[0] > prior[4]
        assert torch.allclose(prior.sum(), torch.tensor(1.0))

    def test_get_prior_clustered(self):
        """Verifies the clustered prior (Negative Binomial) has appropriate shape."""
        n = 10
        r = 2.0
        p = 0.5
        prior = get_prior(n, prior_type="clustered", params={"r": r, "p": p})
        # Negative Binomial with r=2, p=0.5 should have mode around z=1-2
        # Verify it's normalized and has reasonable shape
        assert torch.allclose(prior.sum(), torch.tensor(1.0))
        assert torch.all(prior >= 0.0)

    # --- 2. Integrated Attribution Tests ---

    def test_attribution_normalization(self, mock_dist):
        """Ensures the posterior always sums to 1.0 regardless of prior type."""
        n = 20
        priors = ["flat", "community", "clustered"]
        for p_type in priors:
            post = attribute_source(mock_dist, n, prior_type=p_type)
            assert torch.allclose(post.sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.all(post >= 0.0)

    def test_bayesian_mode_shift(self):
        """
        Verify that the community prior (Poisson with low lambda) shifts the MAP 
        estimate toward a lower number of founders compared to the flat prior.
        """
        # Dist where high founder count is slightly likely under flat prior
        p_dist = torch.tensor([0.2, 0.4, 0.4]) 
        n = 10
        
        post_flat = attribute_source(p_dist, n, prior_type="flat")
        post_community = attribute_source(p_dist, n, prior_type="community", prior_params={"lambda": 1.0})
        
        map_flat = torch.argmax(post_flat).item()
        map_community = torch.argmax(post_community).item()
        
        # Community prior with low lambda should pull the peak toward z=1
        assert map_community <= map_flat

    def test_clustered_prior_dominance(self, mock_dist):
        """
        If we have a very strong clustered prior (Negative Binomial with 
        parameters that favor a specific range), the posterior mode should 
        be influenced by the prior.
        """
        n = 50
        # Use Negative Binomial with r=1, p=0.1 to strongly favor low z values
        # This creates a very peaked distribution around z=1-2
        params = {"r": 1.0, "p": 0.1}
        
        post = attribute_source(mock_dist, n, prior_type="clustered", prior_params=params)
        
        # With r=1, p=0.1, the prior strongly favors z=1, so mode should be at index 0 (z=1)
        assert torch.argmax(post).item() == 0

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
"""
Unit tests for the Structural Fingerprinting task (task_how.py).
"""

import pytest
import torch
from nsb.task_how import compute_how_metrics

class TestTaskHow:
    """
    Validates reproductive number, entropy, and extinction risk calculations.
    """

    def test_deterministic_entropy(self):
        """If p_k is 1.0 for one k and 0 for others, entropy must be 0."""
        # Case: Every person infects exactly 2 people.
        p_dist = torch.tensor([0.0, 0.0, 1.0, 0.0])
        metrics = compute_how_metrics(p_dist)
        assert metrics["entropy"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["r0"] == 2.0

    def test_subcritical_extinction(self):
        """If R0 <= 1, the extinction probability must be 1.0."""
        # Case: R0 = 0.5
        p_dist = torch.tensor([0.6, 0.3, 0.1]) 
        metrics = compute_how_metrics(p_dist)
        assert metrics["r0"] < 1.0
        assert metrics["extinction_prob"] == 1.0

    def test_supercritical_extinction(self):
        """If R0 > 1, extinction probability should be between 0 and 1."""
        # Case: R0 = 1.2
        p_dist = torch.tensor([0.2, 0.4, 0.4]) 
        metrics = compute_how_metrics(p_dist)
        assert metrics["r0"] > 1.0
        assert 0.0 < metrics["extinction_prob"] < 1.0

    def test_entropy_maximized(self):
        """Entropy should be higher for more 'spread out' distributions."""
        p_low_vol = torch.tensor([0.1, 0.8, 0.1])
        p_high_vol = torch.tensor([0.33, 0.33, 0.34])
        
        m_low = compute_how_metrics(p_low_vol)
        m_high = compute_how_metrics(p_high_vol)
        
        assert m_high["entropy"] > m_low["entropy"]

    def test_geometric_closed_form(self):
        """
        Tests against a Geometric distribution where q = 1/R0 (for p_0 > p_1...).
        Specifically, for p_k = (1-p)p^k, q = (1-p)/p if p > 0.5.
        """
        # For R0 = 2.0, we need p = 2/3 in geometric distribution
        # p_k = (1-p) * p^k = (1/3) * (2/3)^k
        # Use enough terms to get R0 close to 2.0
        p = 2/3
        probs = []
        for k in range(8):
            prob = (1-p) * (p**k)
            probs.append(prob)
        probs = torch.tensor(probs)
        probs = probs / probs.sum()  # Normalize after truncation
        
        metrics = compute_how_metrics(probs)
        # For geometric distribution, q should be approximately 1/R0
        # With truncation, R0 will be slightly less than 2.0, and q slightly more than 0.5
        # But should still be in a reasonable range
        assert 0.4 < metrics["extinction_prob"] < 0.6
        assert metrics["r0"] > 1.5  # Should be close to 2.0 but may be less due to truncation
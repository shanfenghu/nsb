"""
Unit tests for the Constrained Neural Stick-Breaking (NSB) process model.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from nsb.constrained_model import ConstrainedNSB

# --- Test Fixtures ---

@pytest.fixture
def heavy_tailed_data() -> np.ndarray:
    """A simple, heavy-tailed dataset for testing."""
    rng = np.random.default_rng(42)
    return rng.negative_binomial(n=2, p=0.1, size=256)

@pytest.fixture
def subcritical_model() -> ConstrainedNSB:
    """An instance of the subcritical constrained model."""
    return ConstrainedNSB(hidden_dim=16, constraint='subcritical')

@pytest.fixture
def critical_model() -> ConstrainedNSB:
    """An instance of the critical constrained model."""
    return ConstrainedNSB(hidden_dim=16, constraint='critical')

# --- Helper Function ---
def get_spectral_radius(model: ConstrainedNSB) -> float:
    """Calculates the spectral radius of the recurrent weight matrix."""
    weight_matrix = model.cell.fc_h.weight
    eigenvalues = torch.linalg.eigvals(weight_matrix.data)
    return torch.max(torch.abs(eigenvalues)).item()

# --- Test Cases ---

def test_initialization_subcritical(subcritical_model: ConstrainedNSB):
    """Tests correct initialization for the subcritical model."""
    assert subcritical_model.constraint == 'subcritical'
    assert not hasattr(subcritical_model.cell.fc_h, 'weight_orig')

def test_initialization_critical(critical_model: ConstrainedNSB):
    """Tests correct initialization for the critical model."""
    assert critical_model.constraint == 'critical'
    assert not hasattr(critical_model.cell.fc_h, 'weight_orig')

def test_initialization_invalid_constraint():
    """Tests that an invalid constraint string raises a ValueError."""
    with pytest.raises(ValueError, match="Constraint must be either 'subcritical' or 'critical'"):
        ConstrainedNSB(constraint='invalid_constraint')

def test_enforce_subcritical_constraint_works():
    """
    A critical test to ensure the SVD projection correctly caps the spectral radius.
    """
    model = ConstrainedNSB(hidden_dim=16, constraint='subcritical')
    
    with torch.no_grad():
        W_h = torch.randn(16, 16) * 2
        model.cell.fc_h.weight.data.copy_(W_h)
    
    assert get_spectral_radius(model) > 1.0
    model._enforce_constraint()
    assert get_spectral_radius(model) <= 0.999 + 1e-6

def test_enforce_critical_constraint_works():
    """
    A new test to ensure the SVD projection works for the critical case.
    """
    model = ConstrainedNSB(hidden_dim=16, constraint='critical')

    with torch.no_grad():
        W_h = torch.randn(16, 16) * 2
        model.cell.fc_h.weight.data.copy_(W_h)

    assert get_spectral_radius(model) > 1.0
    model._enforce_constraint()
    assert get_spectral_radius(model) <= 1.0 + 1e-6

def test_fit_subcritical_maintains_constraint(subcritical_model: ConstrainedNSB, heavy_tailed_data: np.ndarray):
    """
    Tests that after a full training run, the subcritical model's
    spectral radius is less than 1.
    """
    subcritical_model.fit(heavy_tailed_data, epochs=2)
    final_radius = get_spectral_radius(subcritical_model)
    assert final_radius <= 0.999 + 1e-6

def test_fit_critical_maintains_constraint(critical_model: ConstrainedNSB, heavy_tailed_data: np.ndarray):
    """
    Tests that after a full training run, the critical model's
    spectral radius is at most 1.
    """
    critical_model.fit(heavy_tailed_data, epochs=2)
    final_radius = get_spectral_radius(critical_model)
    assert final_radius <= 1.0 + 1e-6

def test_constrained_model_can_still_predict(critical_model: ConstrainedNSB, heavy_tailed_data: np.ndarray):
    """
    Ensures that the constrained models inherit and correctly use the
    inference methods from the parent NSB class.
    """
    critical_model.fit(heavy_tailed_data, epochs=2)
    
    pmf = critical_model.predict_pmf(k_max=10)
    assert pmf.shape == (11,)
    assert np.all(pmf >= 0)
    assert 0.9 < np.sum(pmf) <= 1.0
    
    ll = critical_model.log_likelihood(heavy_tailed_data[:10])
    assert isinstance(ll, float)
    assert np.isfinite(ll)

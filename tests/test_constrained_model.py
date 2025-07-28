"""
Unit tests for the refactored Constrained Neural Stick-Breaking (NSB)
process model.
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

# --- Helper Function ---
def get_spectral_radius(model: ConstrainedNSB) -> float:
    """Calculates the spectral radius of the recurrent weight matrix."""
    weight_matrix = model.cell.fc_h.weight
    eigenvalues = torch.linalg.eigvals(weight_matrix.data)
    return torch.max(torch.abs(eigenvalues)).item()

# --- Test Cases ---

def test_initialization():
    """Tests correct initialization for the constrained model."""
    model = ConstrainedNSB(hidden_dim=16, max_radius=0.9)
    assert model.max_radius == 0.9
    assert isinstance(model.cell, nn.Module)

@pytest.mark.parametrize("invalid_radius", [0, -1.0, "string"])
def test_initialization_invalid_radius(invalid_radius):
    """Tests that an invalid max_radius raises a ValueError."""
    with pytest.raises(ValueError):
        ConstrainedNSB(max_radius=invalid_radius)

@pytest.mark.parametrize("target_radius", [0.5, 0.99, 1.0, 1.5])
def test_enforce_constraint_works(target_radius: float):
    """
    A critical test to ensure the SVD projection correctly caps the spectral
    radius at the target_radius.
    """
    model = ConstrainedNSB(hidden_dim=16, max_radius=target_radius)
    
    # Manually set the weight matrix to have a spectral radius > target_radius + 1
    with torch.no_grad():
        W_h = torch.randn(16, 16) * (target_radius + 2)
        model.cell.fc_h.weight.data.copy_(W_h)
    
    # Ensure the radius is indeed larger before the constraint
    assert get_spectral_radius(model) > target_radius
    
    # Apply the constraint
    model._enforce_constraint()
    
    # Check that the radius is now <= target_radius
    assert get_spectral_radius(model) <= target_radius + 1e-6

@pytest.mark.parametrize("target_radius", [0.8, 1.0])
def test_fit_maintains_constraint(target_radius: float, heavy_tailed_data: np.ndarray):
    """
    Tests that after a full training run, the model's spectral radius
    respects the specified max_radius cap.
    """
    model = ConstrainedNSB(hidden_dim=16, max_radius=target_radius)
    model.fit(heavy_tailed_data, epochs=2)
    final_radius = get_spectral_radius(model)
    assert final_radius <= target_radius + 1e-6

def test_constrained_model_can_still_predict(heavy_tailed_data: np.ndarray):
    """
    Ensures that the constrained model can still be trained and used for
    inference, inheriting correctly from the parent NSB class.
    """
    model = ConstrainedNSB(hidden_dim=16, max_radius=0.99)
    model.fit(heavy_tailed_data, epochs=2)
    
    # Test predict_pmf
    pmf = model.predict_pmf(k_max=10)
    assert pmf.shape == (11,)
    assert np.all(pmf >= 0)
    assert 0.9 < np.sum(pmf) <= 1.0
    
    # Test log_likelihood
    ll = model.log_likelihood(heavy_tailed_data[:10])
    assert isinstance(ll, float)
    assert np.isfinite(ll)

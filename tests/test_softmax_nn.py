"""
Unit tests for the SoftmaxNN baseline model.
"""
import pytest
import numpy as np
import torch

from nsb.softmax_nn import SoftmaxNN

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple, well-behaved dataset for testing."""
    return np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5])

@pytest.fixture
def fitted_model(simple_data: np.ndarray) -> SoftmaxNN:
    """A SoftmaxNN model instance already fitted to simple_data."""
    # k_max is set high enough to include all data
    model = SoftmaxNN(k_max=10, hidden_dim=16)
    # Use fewer epochs for faster testing
    model.fit(simple_data, epochs=10)
    return model

# --- Test Cases ---

def test_initialization():
    """
    Tests that the model initializes correctly with the specified k_max.
    """
    model = SoftmaxNN(k_max=50)
    assert model.k_max == 50
    assert model.pmf_ is None, "pmf_ should be None before fitting."
    assert isinstance(model.model, torch.nn.Module)
    # Check output layer dimension
    output_layer = list(model.model.network.children())[-1]
    assert output_layer.out_features == 51

@pytest.mark.parametrize("invalid_k", [0, -10, 5.5])
def test_initialization_with_invalid_k_max_raises_error(invalid_k):
    """
    Tests that initializing with an invalid k_max raises a ValueError.
    """
    with pytest.raises(ValueError, match="k_max must be a positive integer"):
        SoftmaxNN(k_max=invalid_k)

def test_fit_populates_pmf(simple_data: np.ndarray):
    """
    Tests that the fit method runs and populates the pmf_ attribute.
    """
    model = SoftmaxNN(k_max=10)
    model.fit(simple_data, epochs=2)
    assert model.pmf_ is not None
    assert isinstance(model.pmf_, np.ndarray)

def test_predict_pmf_returns_valid_distribution(fitted_model: SoftmaxNN):
    """
    Tests that the learned PMF is a valid probability distribution.
    """
    pmf = fitted_model.predict_pmf()
    assert pmf.shape == (fitted_model.k_max + 1,)
    assert np.all(pmf >= 0), "All probabilities must be non-negative."
    assert np.sum(pmf) == pytest.approx(1.0), "Probabilities must sum to 1."

def test_predict_pmf_before_fitting_raises_error():
    """
    Tests that calling predict_pmf() before fit() raises a RuntimeError.
    """
    model = SoftmaxNN(k_max=10)
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict_pmf()

def test_log_likelihood_before_fitting_raises_error():
    """
    Tests that calling log_likelihood() before fit() raises a RuntimeError.
    """
    model = SoftmaxNN(k_max=10)
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.log_likelihood(np.array([1, 2]))

def test_log_likelihood_for_data_within_support(fitted_model: SoftmaxNN):
    """
    Tests that log_likelihood calculates a finite value for in-support data.
    """
    test_data = np.array([0, 1, 2, 3])
    ll = fitted_model.log_likelihood(test_data)
    assert np.isfinite(ll)

def test_log_likelihood_for_data_outside_support(fitted_model: SoftmaxNN):
    """
    Tests the critical case where test data is outside the model's support.
    The log-likelihood should be -inf.
    """
    # Create test data where one point is > k_max
    test_data = np.array([1, 2, fitted_model.k_max + 1])
    ll = fitted_model.log_likelihood(test_data)
    assert ll == -np.inf

# --- Tests for Invalid Inputs and Edge Cases ---

def test_fit_with_no_valid_data_raises_error():
    """
    Tests that fit() raises a ValueError if all data is outside the support.
    """
    model = SoftmaxNN(k_max=5)
    data_outside_support = np.array([6, 7, 8])
    with pytest.raises(ValueError, match="No data points are within the specified support"):
        model.fit(data_outside_support)

def test_fit_filters_data_outside_support(simple_data: np.ndarray):
    """
    Tests that fit() correctly ignores data points > k_max during training.
    """
    # Set k_max to 3, so values 4 and 5 from simple_data should be ignored.
    model = SoftmaxNN(k_max=3)
    # This should run without error
    model.fit(simple_data, epochs=2)
    
    # The learned pmf should not have mass > 3
    pmf = model.predict_pmf()
    assert pmf.shape == (4,)
    
    # Check that the log-likelihood of ignored points is -inf
    ll = model.log_likelihood(np.array([4, 5]))
    assert ll == -np.inf

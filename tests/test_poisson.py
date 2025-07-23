"""
Unit tests for the PoissonMLE baseline model.
"""
import pytest
import numpy as np
from scipy.stats import poisson

from nsb.poisson import PoissonMLE

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple, well-behaved dataset for testing."""
    return np.array([1, 2, 3, 4, 5])

@pytest.fixture
def fitted_model(simple_data: np.ndarray) -> PoissonMLE:
    """A PoissonMLE model instance already fitted to simple_data."""
    model = PoissonMLE()
    model.fit(simple_data)
    return model

# --- Test Cases ---

def test_initialization():
    """
    Tests that the model initializes with lambda_ set to None.
    """
    model = PoissonMLE()
    assert model.lambda_ is None, "Model should initialize with lambda_ as None."

def test_fit_calculates_correct_lambda(simple_data: np.ndarray):
    """
    Tests that the fit method correctly calculates the sample mean for lambda.
    """
    model = PoissonMLE()
    model.fit(simple_data)
    
    expected_lambda = np.mean(simple_data) # Should be 3.0
    assert model.lambda_ is not None
    assert model.lambda_ == pytest.approx(expected_lambda)

def test_fit_with_single_value_data():
    """
    Tests fitting with a dataset containing only one unique value.
    """
    data = np.array([5, 5, 5, 5])
    model = PoissonMLE()
    model.fit(data)
    
    assert model.lambda_ == pytest.approx(5.0)

def test_pmf_before_fitting_raises_error():
    """
    Tests that calling pmf() before fit() raises a RuntimeError.
    """
    model = PoissonMLE()
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.pmf(np.array([1, 2]))

def test_log_likelihood_before_fitting_raises_error():
    """
    Tests that calling log_likelihood() before fit() raises a RuntimeError.
    """
    model = PoissonMLE()
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.log_likelihood(np.array([1, 2]))

def test_pmf_returns_correct_probabilities(fitted_model: PoissonMLE):
    """
    Tests that the pmf method returns probabilities consistent with scipy.
    """
    k_values = np.array([0, 1, 2, 3, 4, 5])
    
    # Get probabilities from our model
    model_probs = fitted_model.pmf(k_values)
    
    # Get expected probabilities from scipy's implementation
    expected_probs = poisson.pmf(k_values, mu=fitted_model.lambda_)
    
    np.testing.assert_allclose(model_probs, expected_probs, rtol=1e-6)

def test_log_likelihood_returns_correct_value(fitted_model: PoissonMLE):
    """
    Tests that the log_likelihood method returns the correct average value.
    """
    test_data = np.array([2, 3, 4])
    
    # Get log-likelihood from our model
    model_ll = fitted_model.log_likelihood(test_data)
    
    # Calculate expected log-likelihood manually
    log_probs = poisson.logpmf(test_data, mu=fitted_model.lambda_)
    expected_ll = np.mean(log_probs)
    
    assert model_ll == pytest.approx(expected_ll)

# --- Tests for Invalid Inputs and Edge Cases ---

def test_fit_raises_error_on_empty_data():
    """
    Tests that fit() raises a ValueError when given an empty array.
    """
    model = PoissonMLE()
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        model.fit(np.array([]))

def test_fit_raises_error_on_invalid_input_type():
    """
    Tests that fit() raises a ValueError for non-NumPy array inputs.
    """
    model = PoissonMLE()
    with pytest.raises(ValueError, match="Input data must be a 1D NumPy array"):
        # Pass a list instead of a NumPy array
        model.fit([1, 2, 3])

def test_fit_raises_error_on_multidimensional_array():
    """
    Tests that fit() raises a ValueError for multi-dimensional arrays.
    """
    model = PoissonMLE()
    with pytest.raises(ValueError, match="Input data must be a 1D NumPy array"):
        model.fit(np.array([[1, 2], [3, 4]]))

def test_fit_raises_error_on_non_positive_lambda():
    """
    Tests that fit() raises a ValueError if the data results in a lambda of 0.
    """
    model = PoissonMLE()
    with pytest.raises(ValueError, match="Data results in a non-positive lambda"):
        model.fit(np.array([0, 0, 0]))

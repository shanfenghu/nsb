"""
Unit tests for the NegativeBinomialMLE baseline model.
"""
import pytest
import numpy as np
from scipy.stats import nbinom

from nsb.negative_binomial import NegativeBinomialMLE

# --- Test Fixtures ---

@pytest.fixture
def overdispersed_data() -> np.ndarray:
    """A simple, well-behaved overdispersed dataset."""
    # Mean is approx 5, variance is approx 10.
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])

@pytest.fixture
def underdispersed_data() -> np.ndarray:
    """A dataset that is not overdispersed (variance < mean)."""
    return np.array([4, 5, 5, 6]) # Mean=5, Var=0.5

@pytest.fixture
def fitted_model(overdispersed_data: np.ndarray) -> NegativeBinomialMLE:
    """A NegativeBinomialMLE model instance already fitted to data."""
    model = NegativeBinomialMLE()
    model.fit(overdispersed_data)
    return model

# --- Test Cases ---

def test_initialization():
    """
    Tests that the model initializes with r_ and p_ set to None.
    """
    model = NegativeBinomialMLE()
    assert model.r_ is None
    assert model.p_ is None

def test_fit_populates_parameters(overdispersed_data: np.ndarray):
    """
    Tests that the fit method successfully populates the r_ and p_ parameters.
    """
    model = NegativeBinomialMLE()
    model.fit(overdispersed_data)
    
    assert model.r_ is not None
    assert model.p_ is not None
    assert isinstance(model.r_, float)
    assert isinstance(model.p_, float)
    assert model.r_ > 0
    assert 0 < model.p_ < 1

def test_fitted_moments_are_close_to_data_moments(fitted_model: NegativeBinomialMLE, overdispersed_data: np.ndarray):
    """
    A strong test to verify the fit is reasonable. The mean and variance
    of the fitted distribution should be close to the sample mean and variance.
    Note: MLE does not guarantee a perfect moment match, so a larger
    tolerance is used for variance.
    """
    # Moments of the fitted Negative Binomial distribution
    fitted_mean = fitted_model.r_ * (1 - fitted_model.p_) / fitted_model.p_
    fitted_var = fitted_model.r_ * (1 - fitted_model.p_) / (fitted_model.p_ ** 2)
    
    # Moments of the actual data
    data_mean = np.mean(overdispersed_data)
    data_var = np.var(overdispersed_data)
    
    # Check if the fitted moments are close to the data moments
    assert fitted_mean == pytest.approx(data_mean, rel=1e-3)
    # The MLE variance may not perfectly match the sample variance.
    # We check that it's in a reasonable range by increasing the tolerance.
    assert fitted_var == pytest.approx(data_var, rel=0.2)

def test_pmf_before_fitting_raises_error():
    """
    Tests that calling pmf() before fit() raises a RuntimeError.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.pmf(np.array([1, 2]))

def test_log_likelihood_before_fitting_raises_error():
    """
    Tests that calling log_likelihood() before fit() raises a RuntimeError.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.log_likelihood(np.array([1, 2]))

def test_pmf_returns_correct_probabilities(fitted_model: NegativeBinomialMLE):
    """
    Tests that the pmf method returns probabilities consistent with scipy.
    """
    k_values = np.array([0, 1, 2, 3, 4, 5])
    
    model_probs = fitted_model.pmf(k_values)
    expected_probs = nbinom.pmf(k_values, n=fitted_model.r_, p=fitted_model.p_)
    
    np.testing.assert_allclose(model_probs, expected_probs, rtol=1e-6)

def test_log_likelihood_returns_correct_value(fitted_model: NegativeBinomialMLE):
    """
    Tests that the log_likelihood method returns the correct average value.
    """
    test_data = np.array([2, 3, 4])
    
    model_ll = fitted_model.log_likelihood(test_data)
    
    log_probs = nbinom.logpmf(test_data, n=fitted_model.r_, p=fitted_model.p_)
    expected_ll = np.mean(log_probs)
    
    assert model_ll == pytest.approx(expected_ll)

# --- Tests for Invalid Inputs and Edge Cases ---

def test_fit_raises_error_on_underdispersed_data(underdispersed_data: np.ndarray):
    """
    Tests that fit() raises a ValueError for data that is not overdispersed.
    This is a critical check for the Negative Binomial model.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(ValueError, match="Data variance must be greater than the mean"):
        model.fit(underdispersed_data)

def test_fit_raises_error_on_empty_data():
    """
    Tests that fit() raises a ValueError when given an empty array.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        model.fit(np.array([]))

def test_fit_raises_error_on_invalid_input_type():
    """
    Tests that fit() raises a ValueError for non-NumPy array inputs.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(ValueError, match="Input data must be a 1D NumPy array"):
        model.fit([1, 2, 3, 4, 5, 10]) # Pass a list

def test_fit_raises_error_on_multidimensional_array():
    """
    Tests that fit() raises a ValueError for multi-dimensional arrays.
    """
    model = NegativeBinomialMLE()
    with pytest.raises(ValueError, match="Input data must be a 1D NumPy array"):
        model.fit(np.array([[1, 2], [3, 4]]))

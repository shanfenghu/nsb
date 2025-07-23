"""
Implementation of the Poisson Maximum Likelihood Estimation (MLE) baseline model.
"""
import numpy as np
from scipy.stats import poisson

class PoissonMLE:
    """
    A class to represent the Poisson MLE baseline model for count data.

    This model learns the single parameter `lambda` of a Poisson distribution
    that best fits a given dataset. The Maximum Likelihood Estimate for lambda
    is simply the sample mean of the training data.

    Attributes:
        lambda_ (float | None): The learned rate parameter (lambda) of the
                                Poisson distribution. Initialized to None.
    """

    def __init__(self):
        """Initializes the PoissonMLE model."""
        self.lambda_: float | None = None

    def fit(self, data: np.ndarray):
        """
        Fits the Poisson model to the data using Maximum Likelihood Estimation.

        The MLE for the lambda parameter of a Poisson distribution is the
        sample mean of the data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Raises:
            ValueError: If the input data is not a 1D array or is empty.
            ValueError: If the calculated lambda is not positive.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Input data must be a 1D NumPy array.")
        if data.size == 0:
            raise ValueError("Input data cannot be empty.")

        # The MLE for a Poisson distribution is the sample mean.
        lambda_mle = np.mean(data)

        if lambda_mle <= 0:
            raise ValueError("Data results in a non-positive lambda, which is invalid for a Poisson distribution.")

        self.lambda_ = lambda_mle

    def pmf(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the Probability Mass Function (PMF) for given counts.

        Args:
            k (np.ndarray): A 1D NumPy array of non-negative integer counts for
                            which to calculate the probabilities.

        Returns:
            np.ndarray: An array of probabilities P(Y=k) for each k.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.lambda_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before pmf().")

        return poisson.pmf(k, self.lambda_)

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculates the per-instance average log-likelihood of the data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts
                               (e.g., a test set).

        Returns:
            float: The per-instance average log-likelihood.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.lambda_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before log_likelihood().")

        # Use scipy's logpmf for numerical stability
        log_probs = poisson.logpmf(data, self.lambda_)

        # Return the per-instance average log-likelihood
        return np.mean(log_probs)

"""
Implementation of the Negative Binomial Maximum Likelihood Estimation (MLE)
baseline model.
"""
import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize

class NegativeBinomialMLE:
    """
    A class for the Negative Binomial (NB) MLE baseline model for count data.

    The NB distribution is defined by two parameters: the number of successes `r`
    and the probability of success `p`. Unlike the Poisson model, the MLE for
    these parameters does not have a closed-form solution and must be found
    using numerical optimization.

    This model is particularly useful for overdispersed count data, where the
    variance is greater than the mean.

    Attributes:
        r_ (float | None): The learned dispersion parameter (number of successes, n).
                           Initialized to None.
        p_ (float | None): The learned probability of success parameter (p).
                           Initialized to None.
    """

    def __init__(self):
        """Initializes the NegativeBinomialMLE model."""
        self.r_: float | None = None
        self.p_: float | None = None

    def fit(self, data: np.ndarray):
        """
        Fits the Negative Binomial model to the data using numerical MLE.

        This method uses the method of moments to provide a good initial guess
        for the parameters `r` and `p`, then uses a numerical optimizer
        (L-BFGS-B) to find the parameters that minimize the negative
        log-likelihood of the data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Raises:
            ValueError: If the input data is invalid.
            ValueError: If the data is not overdispersed (variance <= mean),
                        as the NB model is not appropriate in this case.
            RuntimeError: If the numerical optimization fails to converge.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Input data must be a 1D NumPy array.")
        if data.size == 0:
            raise ValueError("Input data cannot be empty.")

        # Method of moments provides a good initial guess for the optimizer.
        mu = np.mean(data)
        var = np.var(data)

        if var <= mu:
            raise ValueError(
                "Data variance must be greater than the mean for a valid "
                "Negative Binomial fit (i.e., data must be overdispersed)."
            )

        p_initial = mu / var
        r_initial = mu**2 / (var - mu)
        initial_params = np.array([r_initial, p_initial])

        # Define the negative log-likelihood function to be minimized.
        def neg_log_likelihood(params):
            r, p = params
            # The optimizer might test invalid parameter values.
            # We return infinity to guide it back to the valid region.
            if r <= 0 or p <= 0 or p >= 1:
                return np.inf
            return -np.sum(nbinom.logpmf(data, n=r, p=p))

        # Set bounds for the parameters: r > 0 and 0 < p < 1.
        bounds = [(1e-6, None), (1e-6, 1 - 1e-6)]

        # Perform the optimization.
        result = minimize(
            neg_log_likelihood,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed to converge: {result.message}")

        self.r_, self.p_ = result.x

    def pmf(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the Probability Mass Function (PMF) for given counts.

        Args:
            k (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Returns:
            np.ndarray: An array of probabilities P(Y=k) for each k.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.r_ is None or self.p_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before pmf().")
        return nbinom.pmf(k, n=self.r_, p=self.p_)

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculates the per-instance average log-likelihood of the data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Returns:
            float: The per-instance average log-likelihood.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.r_ is None or self.p_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before log_likelihood().")
        
        log_probs = nbinom.logpmf(data, n=self.r_, p=self.p_)
        return np.mean(log_probs)

"""
Implementation of the Softmax Neural Network baseline model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class _MLP(nn.Module):
    """
    Internal PyTorch Module for the MLP.
    This defines the neural network architecture.
    """
    def __init__(self, k_max: int, hidden_dim: int):
        super().__init__()
        self.k_max = k_max
        self.network = nn.Sequential(
            # We use a single input neuron as a dummy input
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, k_max + 1)
            # Note: Softmax is not applied here because nn.CrossEntropyLoss,
            # which is used for training, is more numerically stable and
            # expects raw logits as input.
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class SoftmaxNN:
    """
    A neural network baseline that models a count distribution over a fixed,
    finite support {0, 1, ..., k_max} using a softmax output layer.

    This model represents the standard deep learning approach for problems that
    can be framed as classification. It serves as a key baseline to demonstrate
    the limitations of assuming a finite support for count data.

    Attributes:
        k_max (int): The maximum count value the model can represent.
        hidden_dim (int): The number of neurons in the hidden layer.
        model (_MLP): The internal PyTorch neural network model.
        pmf_ (np.ndarray | None): The learned probability mass function over
                                 the range [0, k_max]. Initialized to None.
    """

    def __init__(self, k_max: int, hidden_dim: int = 64):
        """
        Initializes the SoftmaxNN model.

        Args:
            k_max (int): The maximum integer count for the support of the
                         distribution. The model will learn probabilities for
                         counts from 0 to k_max.
            hidden_dim (int, optional): The size of the hidden layer.
                                        Defaults to 64.
        """
        if not isinstance(k_max, int) or k_max <= 0:
            raise ValueError("k_max must be a positive integer.")
        
        self.k_max = k_max
        self.hidden_dim = hidden_dim
        self.model = _MLP(k_max=self.k_max, hidden_dim=self.hidden_dim)
        self.pmf_: np.ndarray | None = None

    def fit(self, data: np.ndarray, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """
        Trains the neural network to learn the probability distribution of the data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.
            epochs (int, optional): The number of training epochs. Defaults to 100.
            lr (float, optional): The learning rate for the Adam optimizer.
                                  Defaults to 1e-3.
            batch_size (int, optional): The size of mini-batches for training.
                                        Defaults to 32.
        """
        # Filter out data points that are outside the model's support
        train_data = data[data <= self.k_max]
        if len(train_data) == 0:
            raise ValueError(f"No data points are within the specified support [0, {self.k_max}].")

        # Convert data to PyTorch tensors. The input to the MLP is a dummy
        # tensor of zeros, as we are learning a single unconditional distribution.
        # The targets are the actual counts, treated as class labels.
        dummy_input = torch.zeros(len(train_data), 1, dtype=torch.float32)
        targets = torch.from_numpy(train_data).long()
        
        dataset = TensorDataset(dummy_input, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for batch_input, batch_targets in loader:
                optimizer.zero_grad()
                logits = self.model(batch_input)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
        
        # After training, store the learned PMF
        self.model.eval()
        with torch.no_grad():
            # Pass a single dummy input to get the full distribution's logits
            final_logits = self.model(torch.zeros(1, 1))
            final_probs = torch.softmax(final_logits, dim=1).squeeze().cpu().numpy()
            self.pmf_ = final_probs

    def predict_pmf(self) -> np.ndarray:
        """
        Returns the learned probability mass function over the support [0, k_max].

        Returns:
            np.ndarray: A 1D NumPy array of size (k_max + 1) where the i-th
                        element is the learned probability P(Y=i).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.pmf_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.pmf_

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculates the per-instance average log-likelihood of the test data.

        For any data point c > k_max, the model assigns a probability of 0,
        resulting in a log-likelihood of -inf.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Returns:
            float: The per-instance average log-likelihood.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.pmf_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        log_probs = []
        for count in data:
            if count > self.k_max:
                # This count is outside the model's support, so its probability is 0.
                log_probs.append(-np.inf)
            else:
                # Use a small epsilon to avoid log(0) if a probability is exactly zero.
                prob = self.pmf_[int(count)]
                log_probs.append(np.log(prob + 1e-9))
        
        return np.mean(log_probs)

"""
Implementation of the Neural Stick-Breaking (NSB) process model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class _NSBCell(nn.Module):
    """
    Internal PyTorch Module for a single recursive step of the NSB process.

    This cell contains the learnable parameters and defines the computation
    to produce the next hidden state and a stick-breaking proportion from the
    previous hidden state.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Layer to update the hidden state
        self.fc_h = nn.Linear(hidden_dim, hidden_dim)
        # Layer to produce the logit for the break proportion
        self.fc_pi = nn.Linear(hidden_dim, 1)

    def forward(self, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the NSB recursion.

        Args:
            h_prev (torch.Tensor): The hidden state from the previous step.
                                   Shape: (batch_size, hidden_dim)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - h_next (torch.Tensor): The next hidden state.
                                         Shape: (batch_size, hidden_dim)
                - pi_logit (torch.Tensor): The logit for the break proportion.
                                           Shape: (batch_size, 1)
        """
        h_next = torch.tanh(self.fc_h(h_prev))
        pi_logit = self.fc_pi(h_next)
        return h_next, pi_logit

class NSB:
    """
    The Neural Stick-Breaking (NSB) process model.

    This model learns a probability distribution over the non-negative integers
    by recursively generating the parameters of a stick-breaking process using
    a recurrent neural network.

    Attributes:
        hidden_dim (int): The dimension of the RNN's hidden state.
        cell (_NSBCell): The internal PyTorch RNN cell.
        h0 (nn.Parameter): The learnable initial hidden state (h_{-1}).
        device (torch.device): The device (CPU or CUDA) the model runs on.
    """
    def __init__(self, hidden_dim: int = 64, init_identity: bool = False):
        """
        Initializes the NSB model.

        Args:
            hidden_dim (int, optional): The size of the RNN's hidden state.
                                        Defaults to 64.
            init_identity (bool, optional): If True, initializes the hidden state
                                           transition matrix W_h as Identity with small
                                           noise. This places the model at the criticality
                                           boundary (ρ=1) from the start, helping it learn
                                           heavy tails more effectively. Defaults to False.
        """
        self.hidden_dim = hidden_dim
        self.cell = _NSBCell(hidden_dim)
        # The initial hidden state is a learnable parameter
        self.h0 = nn.Parameter(torch.randn(1, hidden_dim))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cell.to(self.device)
        self.h0 = self.h0.to(self.device)
        
        # Identity-Centered Initialization (IRNN)
        if init_identity:
            with torch.no_grad():
                # Initialize W_h as Identity
                self.cell.fc_h.weight.copy_(torch.eye(hidden_dim, device=self.device))
                # Add a tiny bit of noise to break symmetry
                self.cell.fc_h.weight.add_(torch.randn_like(self.cell.fc_h.weight) * 0.01)
                # Initialize bias to zero
                self.cell.fc_h.bias.fill_(0)

    def _compute_log_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probabilities for a batch of counts in a numerically
        stable way.

        Args:
            counts (torch.Tensor): A batch of non-negative integer counts.
                                   Shape: (batch_size,)

        Returns:
            torch.Tensor: The log-probabilities for each count in the batch.
                          Shape: (batch_size,)
        """
        batch_size = counts.shape[0]
        k_max = counts.max().item()

        # Expand initial hidden state for the batch
        h = self.h0.expand(batch_size, -1)

        # Store log(1 - pi_k) for each step
        log_one_minus_pis = torch.zeros(batch_size, k_max + 1, device=self.device)
        # Store log(pi_k) for each step
        log_pis = torch.zeros(batch_size, k_max + 1, device=self.device)

        # Unroll the RNN for k_max + 1 steps
        for k in range(k_max + 1):
            h, pi_logit = self.cell(h)
            # Use F.logsigmoid for numerical stability
            # log(pi_k) = log(sigmoid(x)) = logsigmoid(x)
            log_pis[:, k] = F.logsigmoid(pi_logit).squeeze(-1)
            # log(1 - pi_k) = log(1 - sigmoid(x)) = logsigmoid(-x)
            log_one_minus_pis[:, k] = F.logsigmoid(-pi_logit).squeeze(-1)

        # Calculate the log probability for each count c in the batch:
        # log(p_c) = log(pi_c) + sum_{k=0}^{c-1} log(1 - pi_k)
        # We can do this efficiently using a cumulative sum
        cumulative_log_one_minus_pis = torch.cumsum(log_one_minus_pis, dim=1)
        # The sum term is zero for c=0, so we pad it
        sum_term = torch.cat(
            [torch.zeros(batch_size, 1, device=self.device), cumulative_log_one_minus_pis[:, :-1]],
            dim=1
        )
        
        # Gather the final log probabilities for each count in the input batch
        final_log_probs = log_pis.gather(1, counts.unsqueeze(1)).squeeze(-1) + \
                          sum_term.gather(1, counts.unsqueeze(1)).squeeze(-1)

        return final_log_probs

    def fit(self, data: np.ndarray, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """
        Trains the NSB model on the provided data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.
            epochs (int, optional): The number of training epochs. Defaults to 100.
            lr (float, optional): The learning rate for the Adam optimizer. Defaults to 1e-3.
            batch_size (int, optional): The training batch size. Defaults to 32.
        """
        counts_tensor = torch.from_numpy(data).long()
        dataset = TensorDataset(counts_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(list(self.cell.parameters()) + [self.h0], lr=lr)

        self.cell.train()
        for epoch in range(epochs):
            total_loss = 0
            # Use tqdm for a progress bar
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_counts in pbar:
                batch_counts = batch_counts[0].to(self.device)
                optimizer.zero_grad()
                
                log_probs = self._compute_log_probs(batch_counts)
                loss = -log_probs.mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def predict_pmf(self, k_max: int) -> np.ndarray:
        """
        Computes the learned PMF from k=0 to k_max.

        Args:
            k_max (int): The maximum count value to compute the probability for.

        Returns:
            np.ndarray: A 1D NumPy array of size (k_max + 1) containing the PMF.
        """
        self.cell.eval()
        with torch.no_grad():
            h = self.h0
            pmf = np.zeros(k_max + 1)
            remaining_stick = 1.0
            
            for k in range(k_max + 1):
                h, pi_logit = self.cell(h)
                pi_k = torch.sigmoid(pi_logit).item()
                
                pmf[k] = remaining_stick * pi_k
                remaining_stick *= (1 - pi_k)
        
        return pmf

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculates the per-instance average log-likelihood of the test data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Returns:
            float: The per-instance average log-likelihood.
        """
        self.cell.eval()
        log_likelihoods = []
        
        # Process data in batches for efficiency
        counts_tensor = torch.from_numpy(data).long()
        dataset = TensorDataset(counts_tensor)
        loader = DataLoader(dataset, batch_size=128)

        with torch.no_grad():
            for batch_counts in loader:
                batch_counts = batch_counts[0].to(self.device)
                log_probs = self._compute_log_probs(batch_counts)
                log_likelihoods.append(log_probs.cpu().numpy())
        
        return float(np.concatenate(log_likelihoods).mean())

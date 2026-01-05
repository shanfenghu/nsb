"""
Implementation of the Gated Neural Stick-Breaking (NSB-GRU) process model.
"""
import torch
import torch.nn as nn
from nsb.model import NSB

class _NSBGRUCell(nn.Module):
    """
    Internal PyTorch Module for a single recursive step of the NSB-GRU process.

    This cell utilizes a Gated Recurrent Unit (GRU) to update the latent state,
    offering higher expressive capacity than the vanilla RNN cell but at the
    cost of spectral stationarity.
    """
    def __init__(self, hidden_dim: int):
        """
        Initializes the _NSBGRUCell.

        Args:
            hidden_dim (int): The dimension of the latent hidden state.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gated Recurrent Unit transition
        # We use hidden_dim as input_dim because we pass a dummy zero-vector of that size
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # Layer to produce the logit for the break proportion from the updated state
        self.fc_pi = nn.Linear(hidden_dim, 1)

    def forward(self, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the NSB-GRU recursion.

        Since the model is autonomous (modeling an unconditional distribution),
        the GRU receives a dummy zero-vector as input at each step.

        Args:
            h_prev (torch.Tensor): The hidden state from the previous step.
                                   Shape: (batch_size, hidden_dim)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - h_next (torch.Tensor): The next hidden state via GRU update.
                                         Shape: (batch_size, hidden_dim)
                - pi_logit (torch.Tensor): The logit for the break proportion.
                                           Shape: (batch_size, 1)
        """
        # Generate dummy input for the autonomous state transition
        x_dummy = torch.zeros_like(h_prev)
        
        # Update hidden state through the gated manifold
        h_next = self.gru(x_dummy, h_prev)
        
        # Calculate the stick-breaking logit
        pi_logit = self.fc_pi(h_next)
        
        return h_next, pi_logit

class NSBGRU(NSB):
    """
    The Gated Neural Stick-Breaking (NSB-GRU) process model.

    This variant replaces the vanilla first-order recurrence of the NSB-RNN
    with a Gated Recurrent Unit. While it maintains the same generative 
    stick-breaking logic, its non-stationary Jacobian makes it a 'Black-Box' 
    baseline for forensic tasks.

    Attributes:
        hidden_dim (int): The dimension of the RNN's hidden state.
        cell (_NSBGRUCell): The internal PyTorch GRU-based cell.
        h0 (nn.Parameter): The learnable initial hidden state (h_{-1}).
        device (torch.device): The device (CPU or CUDA) the model runs on.
    """
    def __init__(self, hidden_dim: int = 64):
        """
        Initializes the NSBGRU model.

        Args:
            hidden_dim (int, optional): The size of the hidden state.
                                        Defaults to 64.
        """
        # Initialize the base NSB class to set up device and parameters
        super().__init__(hidden_dim)
        
        # Override the cell with the GRU variant
        self.cell = _NSBGRUCell(hidden_dim)
        
        # Re-ensure the cell is on the correct device
        self.cell.to(self.device)
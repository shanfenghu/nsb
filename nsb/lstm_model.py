"""
Implementation of the Long Short-Term Memory Neural Stick-Breaking (NSB-LSTM) process model.
"""
import torch
import torch.nn as nn
from nsb.model import NSB

class _NSBLSTMCell(nn.Module):
    """
    Internal PyTorch Module for a single recursive step of the NSB-LSTM process.

    This cell utilizes an LSTM transition to update the latent state. Unlike 
    the RNN or GRU variants, the LSTM maintains dual latent variables: the 
    hidden state (h) and the cell state (c).
    """
    def __init__(self, hidden_dim: int):
        """
        Initializes the _NSBLSTMCell.

        Args:
            hidden_dim (int): The dimension of the latent hidden and cell states.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # LSTM transition operator
        # Input dimension is hidden_dim to accommodate dummy zero-vectors
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        # Projection from the updated hidden state to the stick-breaking logit
        self.fc_pi = nn.Linear(hidden_dim, 1)

    def forward(self, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one step of the NSB-LSTM recursion.

        Args:
            h_prev (torch.Tensor): The hidden state from the previous step.
                                   Shape: (batch_size, hidden_dim)
            c_prev (torch.Tensor): The cell state from the previous step.
                                   Shape: (batch_size, hidden_dim)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - h_next (torch.Tensor): The updated hidden state.
                - c_next (torch.Tensor): The updated cell state.
                - pi_logit (torch.Tensor): The logit for the break proportion.
        """
        # Autonomous transition: pass zero-vector as input
        x_dummy = torch.zeros_like(h_prev)
        
        # Update states
        h_next, c_next = self.lstm(x_dummy, (h_prev, c_prev))
        
        # Calculate logit from hidden state
        pi_logit = self.fc_pi(h_next)
        
        return h_next, c_next, pi_logit

class NSBLSTM(NSB):
    """
    The Long Short-Term Memory Neural Stick-Breaking (NSB-LSTM) process model.

    This variant utilizes the dual-state memory mechanism of LSTMs. While 
    offering the highest expressive capacity for deep dependencies, its 
    complex gating makes the structural reproduction number (R0) 
    non-stationary and analytically ill-posed.

    Attributes:
        hidden_dim (int): The dimension of the latent states.
        cell (_NSBLSTMCell): The internal LSTM-based cell.
        h0 (nn.Parameter): Learnable initial hidden state.
        c0 (nn.Parameter): Learnable initial cell state.
    """
    def __init__(self, hidden_dim: int = 64):
        """
        Initializes the NSBLSTM model.

        Args:
            hidden_dim (int, optional): Latent state size. Defaults to 64.
        """
        super().__init__(hidden_dim)
        
        # Override cell with LSTM variant
        self.cell = _NSBLSTMCell(hidden_dim)
        
        # LSTM requires an additional learnable initial cell state
        self.c0 = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Ensure all components are on the correct device
        self.cell.to(self.device)
        self.c0 = self.c0.to(self.device)

    def _compute_log_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Overrides the parent method to accommodate the dual-state (h, c) 
        unrolling logic of the LSTM.
        """
        batch_size = counts.shape[0]
        k_max = counts.max().item()

        # Expand initial states for batch
        h = self.h0.expand(batch_size, -1)
        c = self.c0.expand(batch_size, -1)

        import torch.nn.functional as F
        log_one_minus_pis = torch.zeros(batch_size, k_max + 1, device=self.device)
        log_pis = torch.zeros(batch_size, k_max + 1, device=self.device)

        for k in range(k_max + 1):
            h, c, pi_logit = self.cell(h, c)
            log_pis[:, k] = F.logsigmoid(pi_logit).squeeze(-1)
            log_one_minus_pis[:, k] = F.logsigmoid(-pi_logit).squeeze(-1)

        # Standard stick-breaking reconstruction logic
        cumulative_log_one_minus_pis = torch.cumsum(log_one_minus_pis, dim=1)
        sum_term = torch.cat(
            [torch.zeros(batch_size, 1, device=self.device), cumulative_log_one_minus_pis[:, :-1]],
            dim=1
        )
        
        final_log_probs = log_pis.gather(1, counts.unsqueeze(1)).squeeze(-1) + \
                          sum_term.gather(1, counts.unsqueeze(1)).squeeze(-1)

        return final_log_probs

    def predict_pmf(self, k_max: int) -> torch.Tensor:
        """
        Overrides parent to handle (h, c) state unrolling for PMF prediction.
        """
        import numpy as np
        self.cell.eval()
        with torch.no_grad():
            h, c = self.h0, self.c0
            pmf = np.zeros(k_max + 1)
            remaining_stick = 1.0
            
            for k in range(k_max + 1):
                h, c, pi_logit = self.cell(h, c)
                pi_k = torch.sigmoid(pi_logit).item()
                pmf[k] = remaining_stick * pi_k
                remaining_stick *= (1 - pi_k)
        
        return pmf
    
    # Note: fit() and log_likelihood() are inherited and will use the 
    # overridden _compute_log_probs() and predict_pmf() correctly.
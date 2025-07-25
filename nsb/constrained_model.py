"""
Implementation of the Constrained Neural Stick-Breaking (NSB) process model,
designed to control the spectral radius of the recurrent weight matrix.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from nsb.model import NSB, _NSBCell

class ConstrainedNSB(NSB):
    """
    An extension of the NSB model that enforces a constraint on the spectral
    radius of the recurrent weight matrix (W_h) during training.

    Attributes:
        constraint (str): The type of constraint to apply. Can be
                          'subcritical' (rho < 1) or 'critical' (rho = 1).
    """

    def __init__(self, hidden_dim: int = 64, constraint: str = 'subcritical'):
        """
        Initializes the ConstrainedNSB model.

        Args:
            hidden_dim (int, optional): The size of the RNN's hidden state.
                                        Defaults to 64.
            constraint (str, optional): The constraint type. Must be either
                                        'subcritical' or 'critical'.
                                        Defaults to 'subcritical'.
        """
        super().__init__(hidden_dim)
        
        if constraint not in ['subcritical', 'critical']:
            raise ValueError("Constraint must be either 'subcritical' or 'critical'.")
        self.constraint = constraint

    def _enforce_constraint(self):
        """
        Enforces the spectral radius constraint on the recurrent weight matrix W_h
        using Singular Value Decomposition (SVD) projection.
        This is called after each gradient update step.
        """
        if self.constraint == 'subcritical':
            max_sv = 0.8
        elif self.constraint == 'critical':
            max_sv = 1.0
        else:
            return # Should not happen due to __init__ check

        with torch.no_grad():
            W_h = self.cell.fc_h.weight
            # Perform Singular Value Decomposition (SVD)
            U, S, Vh = torch.linalg.svd(W_h, full_matrices=False)
            
            # The singular values are the spectral norms. We cap them.
            S_clamped = torch.clamp(S, max=max_sv)
            
            # Reconstruct the weight matrix with the clamped singular values
            W_h_constrained = U @ torch.diag(S_clamped) @ Vh
            
            # Update the model's weight matrix in-place
            self.cell.fc_h.weight.data.copy_(W_h_constrained)

    def fit(self, data: np.ndarray, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """
        Trains the ConstrainedNSB model, applying the spectral radius
        constraint after each optimization step.

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
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} ({self.constraint})", leave=False)
            for batch_counts in pbar:
                batch_counts = batch_counts[0].to(self.device)
                optimizer.zero_grad()
                
                log_probs = self._compute_log_probs(batch_counts)
                loss = -log_probs.mean()
                
                loss.backward()
                optimizer.step()

                # Enforce the constraint after the optimizer updates the weights.
                self._enforce_constraint()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} ({self.constraint}), Average Loss: {avg_loss:.4f}")
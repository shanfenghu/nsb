"""
Implementation of the Causal Attention Neural Stick-Breaking (NSB-Attention) model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from nsb.model import NSB

class NSBAttention(NSB):
    """
    The Causal Attention Neural Stick-Breaking (NSB-Attention) model.

    This model utilizes a non-recursive, self-attention mechanism to determine 
    the break proportions. While it offers high generative capacity for 
    non-local dependencies, it severs the recursive 'Analytic Bridge' and 
    lacks the spectral stationarity required for stable forensic inversion.

    Attributes:
        hidden_dim (int): The embedding dimension for the attention mechanism.
        pos_encoder (nn.Parameter): Learnable positional encodings for the steps.
        attention (nn.MultiheadAttention): Causal self-attention layer.
        fc_pi (nn.Linear): Projection to stick-breaking logits.
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, max_k: int = 500):
        """
        Initializes the NSBAttention model.

        Args:
            hidden_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            max_k (int): Maximum supported count for positional encodings.
        """
        super().__init__(hidden_dim)
        
        # In Attention, we don't use a recurrent cell or h0 in the traditional sense
        self.cell = None 
        self.h0 = None
        
        self.num_heads = num_heads
        self.max_k = max_k
        
        # Causal Attention layers
        self.pos_encoder = nn.Parameter(torch.randn(max_k + 1, hidden_dim))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc_pi = nn.Linear(hidden_dim, 1)
        
        # Move parameters and modules to device (NSB is not a nn.Module, so we can't use .to())
        self.pos_encoder = self.pos_encoder.to(self.device)
        self.attention.to(self.device)
        self.fc_pi.to(self.device)

    def _get_causal_mask(self, size: int) -> torch.Tensor:
        """Generates a square causal mask to prevent attending to future steps."""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1).bool()
        return mask

    def _compute_all_logits(self, k_limit: int) -> torch.Tensor:
        """
        Computes the stick-breaking logits for all steps up to k_limit 
        using causal self-attention.
        """
        # (L, D) -> (1, L, D) -> (Batch, L, D)
        seq = self.pos_encoder[:k_limit + 1, :].unsqueeze(0)
        
        mask = self._get_causal_mask(k_limit + 1)
        
        # Self-attention with causal masking
        # attn_output shape: (1, L, D)
        attn_output, _ = self.attention(seq, seq, seq, attn_mask=mask, need_weights=False)
        
        # Project to logits: (1, L, 1) -> (L,)
        logits = self.fc_pi(attn_output).squeeze(0).squeeze(-1)
        return logits

    def _compute_log_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Overrides parent to utilize parallel attention computation instead of 
        sequential unrolling.
        """
        batch_size = counts.shape[0]
        k_max = counts.max().item()
        
        if k_max > self.max_k:
            raise ValueError(f"Count {k_max} exceeds max_k {self.max_k} supported by attention.")

        # Compute logits for the whole range [0, k_max] at once
        logits = self._compute_all_logits(k_max)
        
        # Map to probabilities
        log_pis = F.logsigmoid(logits)
        log_one_minus_pis = F.logsigmoid(-logits)

        # Stick-breaking reconstruction logic
        # log(p_k) = log(pi_k) + sum_{i=0}^{k-1} log(1-pi_i)
        cumulative_log_one_minus_pis = torch.cumsum(log_one_minus_pis, dim=0)
        
        # Prepend 0 for the first element's sum term
        sum_term = torch.cat([torch.zeros(1, device=self.device), cumulative_log_one_minus_pis[:-1]])
        
        all_log_probs = log_pis + sum_term
        
        # Gather the log_probs corresponding to the input counts
        return all_log_probs[counts]

    def fit(self, data: np.ndarray, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """
        Trains the NSBAttention model on the provided data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.
            epochs (int, optional): The number of training epochs. Defaults to 100.
            lr (float, optional): The learning rate for the Adam optimizer. Defaults to 1e-3.
            batch_size (int, optional): The training batch size. Defaults to 32.
        """
        counts_tensor = torch.from_numpy(data).long()
        dataset = TensorDataset(counts_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Collect all trainable parameters
        optimizer = optim.Adam(
            list(self.attention.parameters()) + 
            list(self.fc_pi.parameters()) + 
            [self.pos_encoder], 
            lr=lr
        )

        self.attention.train()
        self.fc_pi.train()
        for epoch in range(epochs):
            total_loss = 0
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

    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculates the per-instance average log-likelihood of the test data.

        Args:
            data (np.ndarray): A 1D NumPy array of non-negative integer counts.

        Returns:
            float: The per-instance average log-likelihood.
        """
        self.attention.eval()
        self.fc_pi.eval()
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

    def predict_pmf(self, k_max: int) -> torch.Tensor:
        """Predicts the PMF using the attention-based logits."""
        self.attention.eval()
        self.fc_pi.eval()
        with torch.no_grad():
            logits = self._compute_all_logits(k_max)
            pi_k = torch.sigmoid(logits).cpu().numpy()
            
            pmf = np.zeros(k_max + 1)
            remaining_stick = 1.0
            for k in range(k_max + 1):
                pmf[k] = remaining_stick * pi_k[k]
                remaining_stick *= (1 - pi_k[k])
                
        return pmf
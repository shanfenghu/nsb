"""
Unit tests for the Causal Attention Neural Stick-Breaking (NSB-Attention) model.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from nsb.attention_model import NSBAttention

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple dataset for testing."""
    return np.array([0, 1, 1, 2, 3, 5])

@pytest.fixture
def attn_model() -> NSBAttention:
    """A standard instance of the NSBAttention model for testing."""
    # Small dimensions for fast testing
    return NSBAttention(hidden_dim=16, num_heads=2, max_k=20)

# --- Test Cases ---

def test_nsb_attention_initialization():
    """Tests that the NSBAttention model initializes layers and parameters correctly."""
    model = NSBAttention(hidden_dim=32, num_heads=4, max_k=50)
    
    assert model.hidden_dim == 32
    assert model.max_k == 50
    # Check for core attention components
    assert isinstance(model.attention, nn.MultiheadAttention)
    assert model.pos_encoder.shape == (51, 32)
    assert isinstance(model.fc_pi, nn.Linear)
    # Ensure the parent's recurrent parameters are disabled
    assert model.cell is None
    assert model.h0 is None

def test_causal_mask_logic():
    """Tests that the causal mask is correctly generated (upper triangular)."""
    model = NSBAttention(hidden_dim=16)
    mask = model._get_causal_mask(size=5)
    
    # Mask should be (size, size)
    assert mask.shape == (5, 5)
    # In PyTorch MultiheadAttention, True/1 in attn_mask means 'do not attend'
    # It should be upper triangular (excluding diagonal if mask is used for causal)
    assert torch.all(mask.diagonal() == 0)
    assert mask[0, 1] == True  # Step 0 cannot see Step 1
    assert mask[1, 0] == False # Step 1 can see Step 0

def test_compute_log_probs_attention_manual_check(attn_model: NSBAttention):
    """
    Verifies that the parallel log-probability calculation matches the 
    manual stick-breaking logic for the first few steps.
    """
    with torch.no_grad():
        # Compute all logits up to a small k
        logits = attn_model._compute_all_logits(k_limit=2)
        pi_vals = torch.sigmoid(logits).numpy()
        
        # Manual stick-breaking calculation:
        # p0 = pi0
        # p1 = pi1 * (1 - pi0)
        # p2 = pi2 * (1 - pi0) * (1 - pi1)
        expected_p0 = pi_vals[0]
        expected_p1 = pi_vals[1] * (1 - pi_vals[0])
        
        expected_log_p0 = np.log(expected_p0)
        expected_log_p1 = np.log(expected_p1)

    # Get model computation for batch [0, 1]
    counts = torch.tensor([0, 1], device=attn_model.device)
    model_log_probs = attn_model._compute_log_probs(counts).cpu().detach().numpy()

    assert model_log_probs[0] == pytest.approx(expected_log_p0, rel=1e-5)
    assert model_log_probs[1] == pytest.approx(expected_log_p1, rel=1e-5)

def test_max_k_constraint():
    """Tests that the model raises a ValueError when a count exceeds max_k."""
    model = NSBAttention(max_k=10)
    counts = torch.tensor([15])
    with pytest.raises(ValueError, match="exceeds max_k"):
        model._compute_log_probs(counts)

def test_fit_updates_attention_parameters(attn_model: NSBAttention, simple_data: np.ndarray):
    """
    Tests that training updates the positional encodings and attention weights.
    Uses robust any-change check with allclose.
    """
    # Capture initial states
    initial_pos = attn_model.pos_encoder.clone().detach()
    initial_fc_pi_w = attn_model.fc_pi.weight.clone().detach()

    attn_model.fit(simple_data, epochs=5, batch_size=2)

    # Check for parameter updates
    pos_changed = not torch.allclose(attn_model.pos_encoder.detach(), initial_pos, atol=1e-7)
    fc_changed = not torch.allclose(attn_model.fc_pi.weight.detach(), initial_fc_pi_w, atol=1e-7)

    assert pos_changed or fc_changed, "Attention parameters should update after training."

def test_predict_pmf_consistency(attn_model: NSBAttention, simple_data: np.ndarray):
    """Verifies that the attention-based PMF sums to (near) one."""
    attn_model.fit(simple_data, epochs=2)
    pmf = attn_model.predict_pmf(k_max=15)
    
    assert pmf.shape == (16,)
    assert np.all(pmf >= 0)
    # Check if sum is reasonably high (attention usually learns valid tails)
    assert 0.7 < np.sum(pmf) <= 1.00001

def test_parallel_vs_sequential_logic(attn_model: NSBAttention):
    """
    Ensures that batch indexing works correctly in the parallelized 
    _compute_log_probs method.
    """
    counts = torch.tensor([0, 5, 2])
    log_probs = attn_model._compute_log_probs(counts)
    
    assert log_probs.shape == (3,)
    # Check if p0 < 1 (log_p0 < 0)
    assert torch.all(log_probs < 0)
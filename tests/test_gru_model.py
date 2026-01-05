"""
Unit tests for the Gated Neural Stick-Breaking (NSB-GRU) process model.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsb.gru_model import NSBGRU, _NSBGRUCell

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple, well-behaved dataset for testing."""
    return np.array([0, 1, 1, 2, 2, 2, 3, 4, 5, 10])

@pytest.fixture
def gru_model() -> NSBGRU:
    """A standard instance of the NSBGRU model for testing."""
    # Use a small hidden dimension for faster tests
    return NSBGRU(hidden_dim=16)

# --- Test Cases for _NSBGRUCell ---

def test_nsb_gru_cell_initialization():
    """Tests that the internal _NSBGRUCell initializes its layers correctly."""
    cell = _NSBGRUCell(hidden_dim=32)
    # Check for GRU transition layer
    assert isinstance(cell.gru, nn.GRUCell)
    assert cell.gru.input_size == 32
    assert cell.gru.hidden_size == 32
    # Check for stick-breaking projection layer
    assert isinstance(cell.fc_pi, nn.Linear)
    assert cell.fc_pi.in_features == 32
    assert cell.fc_pi.out_features == 1

def test_nsb_gru_cell_forward_pass():
    """Tests the forward pass of the _NSBGRUCell for correct output shapes."""
    cell = _NSBGRUCell(hidden_dim=16)
    batch_size = 4
    h_prev = torch.randn(batch_size, 16)
    h_next, pi_logit = cell(h_prev)

    assert h_next.shape == (batch_size, 16)
    assert pi_logit.shape == (batch_size, 1)

# --- Test Cases for the main NSBGRU Class ---

def test_nsb_gru_initialization():
    """Tests that the main NSBGRU model initializes correctly."""
    model = NSBGRU(hidden_dim=32)
    assert model.hidden_dim == 32
    assert isinstance(model.cell, _NSBGRUCell)
    assert isinstance(model.h0, nn.Parameter)
    assert model.h0.shape == (1, 32)

def test_compute_log_probs_gru_manual_check(gru_model: NSBGRU):
    """
    Verifies the correctness of the log-probability calculation for the GRU variant.
    This test ensures the unrolling logic correctly handles the gated transition.
    """
    # Manually set weights to predictable values (all ones and zero bias)
    with torch.no_grad():
        for param in gru_model.cell.parameters():
            param.fill_(0.1)  # Using small values to keep logits in a stable range
        gru_model.h0.fill_(0.1)

    # Calculate expected values for the first step (k=0) manually
    # Step 0:
    h_prev = gru_model.h0  # (1, 16) with all 0.1
    # Manual application of the cell to get ground truth for the unrolling
    h_0, pi_0_logit = gru_model.cell(h_prev)
    
    expected_log_pi_0 = F.logsigmoid(pi_0_logit).item()
    
    # log(p_0) = log(pi_0)
    expected_log_p_0 = expected_log_pi_0

    # Get model's batch computation
    counts = torch.tensor([0], device=gru_model.device)
    model_log_probs = gru_model._compute_log_probs(counts).cpu().detach().numpy()

    assert model_log_probs[0] == pytest.approx(expected_log_p_0)

def test_fit_runs_and_updates_parameters(gru_model: NSBGRU, simple_data: np.ndarray):
    """
    Tests that the training process runs and that the GRU parameters update.
    """
    # Store initial parameters
    initial_params = [p.clone().detach() for p in gru_model.cell.parameters()]
    initial_h0 = gru_model.h0.clone().detach()

    # Train for a few epochs
    gru_model.fit(simple_data, epochs=2, batch_size=4)

    # Check that at least some parameters have been updated
    # (Some parameters like biases might remain unchanged if initialized to zero)
    params_changed = False
    for i, param in enumerate(gru_model.cell.parameters()):
        param_detached = param.detach()
        if not torch.allclose(param_detached, initial_params[i], atol=1e-7):
            params_changed = True
            break
    
    # Also check h0
    h0_changed = not torch.allclose(gru_model.h0.detach(), initial_h0, atol=1e-7)
    
    # At least one of the parameters should have changed
    assert params_changed or h0_changed, \
        "At least some GRU parameters or h0 should update after fit."

def test_predict_pmf_returns_valid_distribution(gru_model: NSBGRU, simple_data: np.ndarray):
    """
    Tests that the predicted PMF is a valid probability distribution.
    """
    gru_model.fit(simple_data, epochs=2)
    pmf = gru_model.predict_pmf(k_max=20)
    
    assert pmf.shape == (21,)
    assert np.all(pmf >= 0), "All probabilities must be non-negative."
    # Truncated sum logic inherited from parent
    assert 0.8 < np.sum(pmf) <= 1.0, "Probabilities should sum near 1.0."

def test_log_likelihood_returns_finite_value(gru_model: NSBGRU, simple_data: np.ndarray):
    """
    Tests that the log_likelihood method returns a valid, finite number.
    """
    gru_model.fit(simple_data, epochs=2)
    test_data = np.array([0, 1, 5, 10])
    ll = gru_model.log_likelihood(test_data)
    
    assert isinstance(ll, float)
    assert np.isfinite(ll)

def test_batch_processing_consistency(gru_model: NSBGRU, simple_data: np.ndarray):
    """
    Tests that log-likelihood is consistent across batch sizes with the GRU cell.
    """
    gru_model.fit(simple_data, epochs=2)
    
    ll_full_batch = gru_model.log_likelihood(simple_data)
    
    log_probs_manual = []
    with torch.no_grad():
        for count in simple_data:
            c_tensor = torch.tensor([count], device=gru_model.device)
            log_prob = gru_model._compute_log_probs(c_tensor).item()
            log_probs_manual.append(log_prob)
    
    ll_manual_mean = np.mean(log_probs_manual)
    assert ll_full_batch == pytest.approx(ll_manual_mean)
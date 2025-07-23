"""
Unit tests for the Neural Stick-Breaking (NSB) process model.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsb.model import NSB, _NSBCell

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple, well-behaved dataset for testing."""
    return np.array([0, 1, 1, 2, 2, 2, 3, 4, 5, 10])

@pytest.fixture
def nsb_model() -> NSB:
    """A standard instance of the NSB model for testing."""
    # Use a small hidden dimension for faster tests
    return NSB(hidden_dim=16)

# --- Test Cases for _NSBCell ---

def test_nsb_cell_initialization():
    """Tests that the internal _NSBCell initializes its layers correctly."""
    cell = _NSBCell(hidden_dim=32)
    assert isinstance(cell.fc_h, nn.Linear)
    assert cell.fc_h.in_features == 32
    assert cell.fc_h.out_features == 32
    assert isinstance(cell.fc_pi, nn.Linear)
    assert cell.fc_pi.in_features == 32
    assert cell.fc_pi.out_features == 1

def test_nsb_cell_forward_pass():
    """Tests the forward pass of the _NSBCell for correct output shapes."""
    cell = _NSBCell(hidden_dim=16)
    batch_size = 4
    h_prev = torch.randn(batch_size, 16)
    h_next, pi_logit = cell(h_prev)

    assert h_next.shape == (batch_size, 16)
    assert pi_logit.shape == (batch_size, 1)

# --- Test Cases for the main NSB Class ---

def test_nsb_initialization():
    """Tests that the main NSB model initializes correctly."""
    model = NSB(hidden_dim=32)
    assert model.hidden_dim == 32
    assert isinstance(model.cell, _NSBCell)
    assert isinstance(model.h0, nn.Parameter)
    assert model.h0.shape == (1, 32)

def test_compute_log_probs_manual_check(nsb_model: NSB):
    """
    A critical test to verify the correctness of the log-probability calculation.
    We manually set the weights to predictable values and check the output.
    """
    # Manually set weights to predictable values (all ones and zero bias)
    with torch.no_grad():
        for param in nsb_model.cell.parameters():
            param.fill_(1.0)
        nsb_model.h0.fill_(0.1)

    # Calculate expected values by hand for k=0 and k=1
    # Step 0:
    h_minus_1 = nsb_model.h0 # (1, 16) with all 0.1
    h_0_pre_tanh = nsb_model.cell.fc_h(h_minus_1) # 0.1 * 16 * 1.0 + 1.0 = 2.6
    h_0 = torch.tanh(h_0_pre_tanh)
    pi_0_logit = nsb_model.cell.fc_pi(h_0)
    
    expected_log_pi_0 = F.logsigmoid(pi_0_logit).item()
    expected_log_1_minus_pi_0 = F.logsigmoid(-pi_0_logit).item()
    
    # log(p_0) = log(pi_0)
    expected_log_p_0 = expected_log_pi_0

    # Step 1:
    h_1_pre_tanh = nsb_model.cell.fc_h(h_0)
    h_1 = torch.tanh(h_1_pre_tanh)
    pi_1_logit = nsb_model.cell.fc_pi(h_1)
    expected_log_pi_1 = F.logsigmoid(pi_1_logit).item()

    # log(p_1) = log(pi_1) + log(1 - pi_0)
    expected_log_p_1 = expected_log_pi_1 + expected_log_1_minus_pi_0

    # Get model's computation
    counts = torch.tensor([0, 1], device=nsb_model.device)
    model_log_probs = nsb_model._compute_log_probs(counts).cpu().detach().numpy()

    assert model_log_probs[0] == pytest.approx(expected_log_p_0)
    assert model_log_probs[1] == pytest.approx(expected_log_p_1)

def test_fit_runs_and_updates_parameters(nsb_model: NSB, simple_data: np.ndarray):
    """
    Tests that the training process runs and that the model's parameters change.
    """
    # Store initial parameters
    initial_params = [p.clone() for p in nsb_model.cell.parameters()]
    initial_h0 = nsb_model.h0.clone()

    # Train for a few epochs
    nsb_model.fit(simple_data, epochs=2, batch_size=4)

    # Check that parameters have been updated
    for i, param in enumerate(nsb_model.cell.parameters()):
        assert not torch.equal(param, initial_params[i]), "Model parameters should be updated after fit."
    assert not torch.equal(nsb_model.h0, initial_h0), "Initial hidden state should be updated."

def test_predict_pmf_returns_valid_distribution(nsb_model: NSB, simple_data: np.ndarray):
    """
    Tests that the predicted PMF is a valid probability distribution.
    """
    nsb_model.fit(simple_data, epochs=2)
    pmf = nsb_model.predict_pmf(k_max=20)
    
    assert pmf.shape == (21,)
    assert np.all(pmf >= 0), "All probabilities must be non-negative."
    # The sum will be slightly less than 1 because it's truncated.
    assert 0.9 < np.sum(pmf) <= 1.0, "Probabilities should sum to approximately 1."

def test_log_likelihood_returns_finite_value(nsb_model: NSB, simple_data: np.ndarray):
    """
    Tests that the log_likelihood method returns a valid, finite number.
    """
    nsb_model.fit(simple_data, epochs=2)
    test_data = np.array([0, 1, 5, 10, 20])
    ll = nsb_model.log_likelihood(test_data)
    
    assert isinstance(ll, float)
    assert np.isfinite(ll)

def test_batch_processing_consistency(nsb_model: NSB, simple_data: np.ndarray):
    """
    Tests that log-likelihood is consistent regardless of batch size.
    """
    nsb_model.fit(simple_data, epochs=2)
    
    # Calculate with a single large batch
    ll_full_batch = nsb_model.log_likelihood(simple_data)
    
    # Manually calculate with smaller batches
    log_probs_manual = []
    with torch.no_grad():
        for count in simple_data:
            c_tensor = torch.tensor([count], device=nsb_model.device)
            log_prob = nsb_model._compute_log_probs(c_tensor).item()
            log_probs_manual.append(log_prob)
    
    ll_manual_mean = np.mean(log_probs_manual)
    
    assert ll_full_batch == pytest.approx(ll_manual_mean)

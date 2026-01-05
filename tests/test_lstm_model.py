"""
Unit tests for the Long Short-Term Memory Neural Stick-Breaking (NSB-LSTM) process model.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsb.lstm_model import NSBLSTM, _NSBLSTMCell

# --- Test Fixtures ---

@pytest.fixture
def simple_data() -> np.ndarray:
    """A simple, well-behaved dataset for testing."""
    return np.array([0, 1, 1, 2, 2, 2, 3, 4, 5, 10])

@pytest.fixture
def lstm_model() -> NSBLSTM:
    """A standard instance of the NSBLSTM model for testing."""
    return NSBLSTM(hidden_dim=16)

# --- Test Cases for _NSBLSTMCell ---

def test_nsb_lstm_cell_initialization():
    """Tests that the internal _NSBLSTMCell initializes its layers correctly."""
    cell = _NSBLSTMCell(hidden_dim=32)
    # Check for LSTM transition layer
    assert isinstance(cell.lstm, nn.LSTMCell)
    assert cell.lstm.input_size == 32
    assert cell.lstm.hidden_size == 32
    # Check for stick-breaking projection layer
    assert isinstance(cell.fc_pi, nn.Linear)
    assert cell.fc_pi.out_features == 1

def test_nsb_lstm_cell_forward_pass():
    """Tests the forward pass of the _NSBLSTMCell for correct output shapes."""
    cell = _NSBLSTMCell(hidden_dim=16)
    batch_size = 4
    h_prev = torch.randn(batch_size, 16)
    c_prev = torch.randn(batch_size, 16)
    h_next, c_next, pi_logit = cell(h_prev, c_prev)

    assert h_next.shape == (batch_size, 16)
    assert c_next.shape == (batch_size, 16)
    assert pi_logit.shape == (batch_size, 1)

# --- Test Cases for the main NSBLSTM Class ---

def test_nsb_lstm_initialization():
    """Tests that the main NSBLSTM model initializes correctly with dual initial states."""
    model = NSBLSTM(hidden_dim=32)
    assert model.hidden_dim == 32
    assert isinstance(model.cell, _NSBLSTMCell)
    assert isinstance(model.h0, nn.Parameter)
    assert isinstance(model.c0, nn.Parameter)
    assert model.h0.shape == (1, 32)
    assert model.c0.shape == (1, 32)

def test_compute_log_probs_lstm_manual_check(lstm_model: NSBLSTM):
    """
    Verifies the log-probability calculation for the LSTM variant.
    Ensures that the overridden unrolling logic correctly handles dual states.
    """
    with torch.no_grad():
        for param in lstm_model.cell.parameters():
            param.fill_(0.1)
        lstm_model.h0.fill_(0.1)
        lstm_model.c0.fill_(0.1)

    # Manual step to get ground truth
    h_0, c_0, pi_0_logit = lstm_model.cell(lstm_model.h0, lstm_model.c0)
    expected_log_p_0 = F.logsigmoid(pi_0_logit).item()

    # Get model computation
    counts = torch.tensor([0], device=lstm_model.device)
    model_log_probs = lstm_model._compute_log_probs(counts).cpu().detach().numpy()

    assert model_log_probs[0] == pytest.approx(expected_log_p_0)

def test_fit_runs_and_updates_parameters(lstm_model: NSBLSTM, simple_data: np.ndarray):
    """
    Tests that training updates the LSTM parameters, h0, and c0.
    Uses robust any-change check with allclose.
    """
    initial_params = [p.clone().detach() for p in lstm_model.cell.parameters()]
    initial_h0 = lstm_model.h0.clone().detach()
    initial_c0 = lstm_model.c0.clone().detach()

    lstm_model.fit(simple_data, epochs=2, batch_size=4)

    # Check if any cell parameters changed
    cell_changed = any(not torch.allclose(p.detach(), initial_params[i], atol=1e-7) 
                       for i, p in enumerate(lstm_model.cell.parameters()))
    
    h0_changed = not torch.allclose(lstm_model.h0.detach(), initial_h0, atol=1e-7)
    c0_changed = not torch.allclose(lstm_model.c0.detach(), initial_c0, atol=1e-7)

    assert cell_changed or h0_changed or c0_changed, \
        "Parameters (cell, h0, or c0) should update after training."

def test_predict_pmf_returns_valid_distribution(lstm_model: NSBLSTM, simple_data: np.ndarray):
    """Verifies that the predicted PMF respects probability constraints."""
    lstm_model.fit(simple_data, epochs=2)
    pmf = lstm_model.predict_pmf(k_max=20)
    
    assert pmf.shape == (21,)
    assert np.all(pmf >= 0)
    assert 0.8 < np.sum(pmf) <= 1.0

def test_log_likelihood_consistency(lstm_model: NSBLSTM, simple_data: np.ndarray):
    """Tests batch processing consistency for the dual-state LSTM."""
    lstm_model.fit(simple_data, epochs=2)
    ll_full = lstm_model.log_likelihood(simple_data)
    
    log_probs_manual = []
    with torch.no_grad():
        for count in simple_data:
            c_tensor = torch.tensor([count], device=lstm_model.device)
            log_probs_manual.append(lstm_model._compute_log_probs(c_tensor).item())
    
    assert ll_full == pytest.approx(np.mean(log_probs_manual))
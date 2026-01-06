"""
Script to count parameters in NSB model and find matching hidden dimensions
for other models (SoftmaxNN, NSBGRU, NSBLSTM, NSBAttention) that have the
closest parameter count.
"""
import torch
import torch.nn as nn
from nsb.model import NSB
from nsb.gru_model import NSBGRU
from nsb.lstm_model import NSBLSTM
from nsb.attention_model import NSBAttention
from nsb.softmax_nn import SoftmaxNN
from nsb.negative_binomial import NegativeBinomialMLE
from nsb.poisson import PoissonMLE


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: A PyTorch model or NSB model instance
        
    Returns:
        int: Total number of trainable parameters
    """
    # Check NSBAttention first since it inherits from NSB
    if isinstance(model, NSBAttention):
        # For NSBAttention, count attention, fc_pi, and pos_encoder
        attention_params = sum(p.numel() for p in model.attention.parameters())
        fc_pi_params = sum(p.numel() for p in model.fc_pi.parameters())
        pos_encoder_params = model.pos_encoder.numel()
        return attention_params + fc_pi_params + pos_encoder_params
    
    elif isinstance(model, (NSB, NSBGRU, NSBLSTM)):
        # For NSB-based models, count cell parameters + h0
        if model.cell is not None:
            cell_params = sum(p.numel() for p in model.cell.parameters())
        else:
            cell_params = 0
        
        if model.h0 is not None:
            h0_params = model.h0.numel()
        else:
            h0_params = 0
        
        # FIX: Include the initial cell state for LSTM
        c0_params = 0
        if hasattr(model, 'c0') and model.c0 is not None:
            c0_params = model.c0.numel()
        
        return cell_params + h0_params + c0_params
    
    elif isinstance(model, SoftmaxNN):
        # For SoftmaxNN, count model parameters
        return sum(p.numel() for p in model.model.parameters())
    
    elif isinstance(model, NegativeBinomialMLE):
        # Negative Binomial has 2 parameters: r_ and p_
        return 2
    
    elif isinstance(model, PoissonMLE):
        # Poisson has 1 parameter: lambda_
        return 1
    
    else:
        # Fallback for standard PyTorch modules
        try:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except AttributeError:
            # If model doesn't have parameters() method, return 0
            return 0


def find_best_hidden_dim(model_class, target_params, model_kwargs=None, search_range=(1, 512)):
    """
    Find the hidden dimension that gives the closest parameter count to target.
    Prefers parameter counts >= target to avoid reviewers claiming we use fewer parameters.
    
    Args:
        model_class: The model class to instantiate
        target_params: Target number of parameters
        model_kwargs: Additional keyword arguments for model initialization
        search_range: Tuple of (min, max) hidden dimensions to search
        
    Returns:
        tuple: (best_hidden_dim, actual_param_count, difference)
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    best_hidden_dim = None
    best_diff = float('inf')
    best_param_count = None
    best_is_above = False  # Track if best match is >= target
    
    # Binary search for efficiency
    min_dim, max_dim = search_range
    
    def compute_penalized_diff(param_count, target):
        """
        Compute difference with penalty for being below target.
        This ensures we prefer matches >= target to avoid reviewers claiming
        we use fewer parameters for baselines.
        """
        diff = abs(param_count - target)
        if param_count < target:
            # Heavily penalize being below target - add a large penalty
            # that ensures we prefer any match >= target over matches below
            # Use a penalty larger than any reasonable difference we'd encounter
            return diff + 10000  # Large penalty to strongly prefer >= target
        return diff
    
    # First, do a coarse search
    for hidden_dim in range(min_dim, max_dim + 1, 4):
        try:
            model = model_class(hidden_dim=hidden_dim, **model_kwargs)
            param_count = count_parameters(model)
            diff = compute_penalized_diff(param_count, target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_hidden_dim = hidden_dim
                best_param_count = param_count
                best_is_above = param_count >= target_params
        except Exception as e:
            # Skip dimensions that cause errors
            continue
    
    # Then refine around the best found dimension
    if best_hidden_dim is not None:
        refine_range = range(max(min_dim, best_hidden_dim - 3), 
                           min(max_dim + 1, best_hidden_dim + 4))
        for hidden_dim in refine_range:
            if hidden_dim == best_hidden_dim:
                continue
            try:
                model = model_class(hidden_dim=hidden_dim, **model_kwargs)
                param_count = count_parameters(model)
                diff = compute_penalized_diff(param_count, target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best_hidden_dim = hidden_dim
                    best_param_count = param_count
                    best_is_above = param_count >= target_params
            except Exception as e:
                continue
    
    # Return the actual absolute difference (not the penalized one)
    actual_diff = abs(best_param_count - target_params) if best_param_count is not None else float('inf')
    return best_hidden_dim, best_param_count, actual_diff


def main(hidden_dim=64):
    """
    Main function to count NSB parameters and find matching hidden dimensions.
    
    Args:
        hidden_dim: Hidden dimension for the NSB model (default: 64)
    """
    print(f"Counting parameters for NSB model with hidden_dim={hidden_dim}...")
    
    # Count parameters in NSB model
    nsb_model = NSB(hidden_dim=hidden_dim)
    nsb_params = count_parameters(nsb_model)
    print(f"\nNSB model parameters: {nsb_params:,}")
    print(f"  - Cell parameters: {sum(p.numel() for p in nsb_model.cell.parameters()):,}")
    print(f"  - h0 parameters: {nsb_model.h0.numel():,}")
    
    print(f"\nFinding best hidden dimensions for other models to match {nsb_params:,} parameters...")
    print("-" * 70)
    
    # Find best hidden_dim for NSBGRU
    print("\nNSBGRU:")
    best_gru_dim, gru_params, gru_diff = find_best_hidden_dim(NSBGRU, nsb_params)
    if best_gru_dim is not None:
        print(f"  Best hidden_dim: {best_gru_dim}")
        print(f"  Parameter count: {gru_params:,} (difference: {gru_diff:,})")
    else:
        print("  Could not find matching hidden dimension")
    
    # Find best hidden_dim for NSBLSTM
    print("\nNSBLSTM:")
    best_lstm_dim, lstm_params, lstm_diff = find_best_hidden_dim(NSBLSTM, nsb_params)
    if best_lstm_dim is not None:
        print(f"  Best hidden_dim: {best_lstm_dim}")
        print(f"  Parameter count: {lstm_params:,} (difference: {lstm_diff:,})")
    else:
        print("  Could not find matching hidden dimension")
    
    # Find best hidden_dim for NSBAttention
    print("\nNSBAttention:")
    # NSBAttention needs num_heads and max_k
    # Use max_k=150 to match SoftmaxNN's k_max=150 for fair comparison
    # Note: hidden_dim must be divisible by num_heads
    best_attn_dim = None
    best_attn_params = None
    best_attn_diff = float('inf')
    best_num_heads = None
    max_k = 150  # Match SoftmaxNN's k_max for fair comparison
    
    # Try different num_heads values with max_k=150
    for num_heads in [1, 2, 4]:
        # Only search hidden_dims that are divisible by num_heads
        min_dim = max(1, num_heads)
        max_dim = 128
        # Create a custom search that only tries multiples of num_heads
        for test_hidden_dim in range(min_dim, max_dim + 1, num_heads):
            try:
                model = NSBAttention(hidden_dim=test_hidden_dim, num_heads=num_heads, max_k=max_k)
                param_count = count_parameters(model)
                diff = abs(param_count - nsb_params)
                
                if diff < best_attn_diff:
                    best_attn_diff = diff
                    best_attn_dim = test_hidden_dim
                    best_attn_params = param_count
                    best_num_heads = num_heads
            except Exception:
                continue
    
    if best_attn_dim is not None:
        print(f"  Best hidden_dim: {best_attn_dim} (with max_k={max_k}, num_heads={best_num_heads})")
        print(f"  Parameter count: {best_attn_params:,} (difference: {best_attn_diff:,})")
        print(f"  (with max_k=150 to match SoftmaxNN)")
    else:
        print("  Could not find matching hidden dimension")
    
    # Find best hidden_dim for SoftmaxNN
    print("\nSoftmaxNN:")
    # SoftmaxNN needs k_max, use a reasonable default (150)
    best_softmax_dim, softmax_params, softmax_diff = find_best_hidden_dim(
        SoftmaxNN,
        nsb_params,
        model_kwargs={'k_max': 150}
    )
    if best_softmax_dim is not None:
        print(f"  Best hidden_dim: {best_softmax_dim}")
        print(f"  Parameter count: {softmax_params:,} (difference: {softmax_diff:,})")
        print(f"  (with k_max=150)")
    else:
        print("  Could not find matching hidden dimension")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  NSB (hidden_dim={hidden_dim}): {nsb_params:,} parameters")
    if best_gru_dim is not None:
        print(f"  NSBGRU (hidden_dim={best_gru_dim}): {gru_params:,} parameters")
    if best_lstm_dim is not None:
        print(f"  NSBLSTM (hidden_dim={best_lstm_dim}): {lstm_params:,} parameters")
    if best_attn_dim is not None:
        print(f"  NSBAttention (hidden_dim={best_attn_dim}, max_k=150, num_heads={best_num_heads}): {best_attn_params:,} parameters")
    if best_softmax_dim is not None:
        print(f"  SoftmaxNN (hidden_dim={best_softmax_dim}): {softmax_params:,} parameters")
    
    print("\nBest hidden dimensions to match NSB parameter count:")
    if best_gru_dim is not None:
        print(f"  NSBGRU: hidden_dim = {best_gru_dim}")
    if best_lstm_dim is not None:
        print(f"  NSBLSTM: hidden_dim = {best_lstm_dim}")
    if best_attn_dim is not None:
        print(f"  NSBAttention: hidden_dim = {best_attn_dim} (max_k = 150, num_heads = {best_num_heads})")
    if best_softmax_dim is not None:
        print(f"  SoftmaxNN: hidden_dim = {best_softmax_dim}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count parameters and find matching hidden dimensions")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for NSB model (default: 64)"
    )
    
    args = parser.parse_args()
    main(hidden_dim=args.hidden_dim)


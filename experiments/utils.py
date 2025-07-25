"""
General utility functions for the experimental scripts.
"""
import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
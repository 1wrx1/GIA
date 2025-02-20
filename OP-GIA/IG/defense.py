"""
Defense methods.

Including:
- Additive noise
- Gradient compression
"""
import numpy as np
import torch


def additive_noise(input_gradient, var=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), var * torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None] * len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient

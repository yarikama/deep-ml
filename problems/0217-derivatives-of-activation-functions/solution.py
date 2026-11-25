import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here
    sigmoid = float(1 / (1 + np.exp(-x)))
    tanh = float((np.exp(2 * x) - 1) / (np.exp(2 * x) + 1))

    return {
        "sigmoid": sigmoid * (1 - sigmoid),
        "tanh": 1 - tanh ** 2,
        "relu": 1.0 if x > 0 else 0.0
    }
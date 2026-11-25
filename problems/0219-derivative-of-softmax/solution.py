import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    x = np.asarray(x, dtype=float)

    exp_x = np.exp(x)
    softmax = exp_x / np.sum(exp_x)

    J = np.diag(softmax) - np.outer(softmax, softmax)

    return J.tolist()


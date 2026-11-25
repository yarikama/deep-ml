import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    # Your code here
    n, m = len(x), len(f(x))
    J = np.zeros((m, n))

    for j in range(n):
        x_add_h, x_sub_h = list(x), list(x)
        
        x_add_h[j] += h
        x_sub_h[j] -= h

        f_add_h = np.array(f(x_add_h), dtype=float)
        f_sub_h = np.array(f(x_sub_h), dtype=float)

        J[:,j] = (f_add_h - f_sub_h) / (2 * h)

    return J.tolist()
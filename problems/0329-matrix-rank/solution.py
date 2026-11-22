import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    if A.size == 0:
        return 0
    
    singular_values = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(singular_values > tol))

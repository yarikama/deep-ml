import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    arr = np.asarray(arr, dtype=float)
    v = arr.ravel()

    if norm_type == "l1":
        return float(np.linalg.norm(v, 1))
    elif norm_type == "l2":
        return float(np.linalg.norm(v))
    elif norm_type == "linf":
        return float(np.linalg.norm(v, np.inf))
    elif norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError("Frobenius norm requires a 2D array")
        return float(np.linalg.norm(arr, 'fro'))

    raise ValueError("Unknown norm_type: {norm_type}")









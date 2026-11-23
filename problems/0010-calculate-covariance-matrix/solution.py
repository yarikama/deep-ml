import numpy as np

def calculate_covariance_matrix(
    vectors: list[list[float]]
) -> list[list[float]]:
    X = np.array(vectors)
    N = X.shape[1]
    Z = X - X.mean(axis=1, keepdims=True)
    cov = Z @ Z.T / (N - 1)
    return cov.tolist()
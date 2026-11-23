import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Your code here
    if Y is None:
        Y = X
    n = X.shape[0]

    X_diff = X - X.mean(axis=0)
    Y_diff = Y - Y.mean(axis=0)
    cov = X_diff.T @ Y_diff / n

    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    return cov / np.outer(std_x, std_y)

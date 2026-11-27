import numpy as np

def train_neuron(
    features: np.ndarray, 
    labels: np.ndarray, 
    initial_weights: np.ndarray, 
    initial_bias: float, 
    learning_rate: float, 
    epochs: int
) -> (np.ndarray, float, list[float]):
    # np all the inputs
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    # size of samples
    n = len(y)
    mse_values = []

    for _ in range(epochs):
        # foward
        z = X @ weights + bias
        predictions = 1.0 / (1.0 + np.exp(-z))

        # foward MSE
        errors = predictions - y
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))

        # backward
        dz = (2.0 / n) * errors * predictions * (1.0 - predictions)
        dw = X.T @ dz
        db = np.sum(dz)

        # update weights
        weights -= learning_rate * dw
        bias -= learning_rate * db

    updated_weights, updated_bias = np.round(weights, 4), round(float(bias), 4)

    return updated_weights, updated_bias, mse_values
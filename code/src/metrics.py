import numpy as np


def rmse(p_true: np.ndarray, p_prime: np.ndarray) -> float:
    """Root Mean Squared Error between observed and reconstructed proportions."""
    diff = p_true - p_prime
    return float(np.sqrt(np.mean(diff ** 2)))


def squared_loss(p_true: np.ndarray, p_prime: np.ndarray) -> float:
    """Sum of squared differences (used for grid search)."""
    return float(np.sum((p_true - p_prime) ** 2))

import numpy as np


def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rowsum > eps, m / np.where(rowsum > eps, rowsum, 1.0), 0.0)
    return out


def build_transition_P(
    C_history: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Build the transition probability matrix P (N+1, N+1) from the observed
    occupancy history C_history (T x N) and the distance matrix.

    Node 0 = outside (outside the venue), nodes 1..N = internal zones.

    Parameters
    ----------
    C_history : ndarray (T, N)
        Observed occupancy at each time bin / internal zone (see data/README.md)
    distances : ndarray (N+1, N+1)
        Normalized distance matrix
    alpha : float
        Distance-decay coefficient
    beta : float
        Destination-popularity exponent

    Returns
    -------
    P : ndarray (N+1, N+1)
        Row-normalized transition probability matrix
    """
    T, N = C_history.shape
    Np = N + 1
    if distances.shape != (Np, Np):
        raise ValueError(f"distances must be ({Np},{Np}), got {distances.shape}")

    p_hist = _safe_row_normalize(C_history)

    prev_sum = C_history[-2].sum() if T >= 2 else C_history[-1].sum()
    cur_sum = C_history[-1].sum()
    denom = prev_sum if prev_sum > 0 else 1.0
    delta_plus = max(cur_sum - prev_sum, 0.0) / denom

    prev_idx = -2 if T >= 2 else -1
    p_prev = np.concatenate(([delta_plus], p_hist[prev_idx] * (1.0 - delta_plus)))

    W = np.exp(-alpha * distances) * (p_prev[None, :] ** beta)
    P = _safe_row_normalize(W)
    return P

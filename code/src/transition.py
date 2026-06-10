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
    観測人数履歴 C_history (T×N) と距離行列から遷移確率行列 P (N+1, N+1) を構築する。

    ノード0 = 外部（会場外）、ノード1..N = 内部ゾーン。

    Parameters
    ----------
    C_history : ndarray (T, N)
        各時刻・各内部ゾーンの観測人数
    distances : ndarray (N+1, N+1)
        正規化済み距離行列
    alpha : float
        距離減衰係数
    beta : float
        目的地人気度の指数

    Returns
    -------
    P : ndarray (N+1, N+1)
        行正規化された遷移確率行列
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

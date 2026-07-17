import numpy as np


def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rowsum > eps, m / np.where(rowsum > eps, rowsum, 1.0), 0.0)


def reconstruct_proportions(
    traj_list: list[list[int]],
    T_steps: int,
    N: int,
) -> np.ndarray:
    """
    Reconstruct the internal-zone occupancy proportions p' (T x N) at each
    time step from a list of trajectories.

    The outside node (0) is not counted. Internal zones 1..N map to
    column indices 0..N-1.
    """
    if not traj_list:
        return np.zeros((T_steps, N), dtype=float)

    counts = np.zeros((T_steps, N), dtype=float)
    for traj in traj_list:
        for t in range(T_steps):
            u = traj[t]
            if 1 <= u <= N:
                counts[t, u - 1] += 1.0

    return counts / float(len(traj_list))


def build_flow_matrix(
    traj_list: list[list[int]],
    T_steps: int,
    Np: int,
) -> np.ndarray:
    """
    Build the zone-to-zone flow matrix F (Np x Np) from a list of
    trajectories (row-normalized).
    """
    F_counts = np.zeros((Np, Np), dtype=float)
    for traj in traj_list:
        for t in range(T_steps - 1):
            F_counts[traj[t], traj[t + 1]] += 1.0
    return _safe_row_normalize(F_counts)

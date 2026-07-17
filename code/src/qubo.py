import numpy as np

Qubo = dict[tuple[int, int], float]


def build_qubo(
    P: np.ndarray,
    T_steps: int,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
) -> Qubo:
    """
    Build a QUBO dict from the transition probability matrix P.

    Variable index: x_{i,t} -> i * T_steps + t
      i=0: outside node, i=1..N: internal zones

    Each term is normalized by its max_abs value, then scaled by lambda and summed.

    Parameters
    ----------
    P : ndarray (N+1, N+1)
        Transition probability matrix
    T_steps : int
        Number of time steps
    lambda_* : float
        Weight coefficient for each term

    Returns
    -------
    Q : dict
        QUBO dict (upper triangle + diagonal terms only)
    """
    Np = P.shape[0]
    Q: Qubo = {}

    def idx(i: int, t: int) -> int:
        return i * T_steps + t

    def add_normalized(base: Qubo, lam: float) -> None:
        if not base:
            return
        max_abs = max(abs(v) for v in base.values()) or 1.0
        for key, val in base.items():
            Q[key] = Q.get(key, 0.0) + lam * (val / max_abs)

    # --- One-hot: penalize multiple selections at the same time step ---
    onehot: Qubo = {}
    for t in range(T_steps):
        for i in range(Np):
            for j in range(i + 1, Np):
                k = (idx(i, t), idx(j, t))
                onehot[k] = onehot.get(k, 0.0) + 2.0
    add_normalized(onehot, lambda_onehot)

    # --- P preference: -P[i,j] for i@t-1 -> j@t ---
    pref: Qubo = {}
    for t in range(1, T_steps):
        for i in range(Np):
            for j in range(Np):
                k = (idx(i, t - 1), idx(j, t))
                pref[k] = pref.get(k, 0.0) - P[i, j]
    add_normalized(pref, lambda_P)

    # --- Visit dispersion: suppress bias toward any particular zone ---
    avg_visits = T_steps / Np
    div: Qubo = {}
    for i in range(Np):
        for t in range(T_steps):
            ii = idx(i, t)
            div[(ii, ii)] = div.get((ii, ii), 0.0) + (1 - 2 * avg_visits)
            for t2 in range(t + 1, T_steps):
                jj = idx(i, t2)
                div[(ii, jj)] = div.get((ii, jj), 0.0) + 2.0
    add_normalized(div, lambda_div)

    # --- Suppress toggling in/out of the outside node (0) ---
    entry: Qubo = {}
    for t in range(T_steps - 1):
        ii, jj = idx(0, t), idx(0, t + 1)
        entry[(ii, ii)] = entry.get((ii, ii), 0.0) + 1.0
        entry[(jj, jj)] = entry.get((jj, jj), 0.0) + 1.0
        entry[(ii, jj)] = entry.get((ii, jj), 0.0) - 2.0
    add_normalized(entry, lambda_entry)

    # --- Smoothing of internal moves ---
    move: Qubo = {}
    for i in range(1, Np):
        for t in range(T_steps - 1):
            ii, jj = idx(i, t), idx(i, t + 1)
            move[(ii, ii)] = move.get((ii, ii), 0.0) + 1.0
            move[(jj, jj)] = move.get((jj, jj), 0.0) + 1.0
            move[(ii, jj)] = move.get((ii, jj), 0.0) - 2.0
    add_normalized(move, lambda_move)

    return Q

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
    遷移確率行列 P から QUBO 辞書を構築する。

    変数インデックス: x_{i,t} → i * T_steps + t
      i=0: 外部ノード, i=1..N: 内部ゾーン

    各項は max_abs で正規化してから λ 倍して合算する。

    Parameters
    ----------
    P : ndarray (N+1, N+1)
        遷移確率行列
    T_steps : int
        時間ステップ数
    lambda_* : float
        各項の重み係数

    Returns
    -------
    Q : dict
        QUBO 辞書（上三角 + 対角成分のみ）
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

    # --- One-Hot: 同時刻で複数選択を罰 ---
    onehot: Qubo = {}
    for t in range(T_steps):
        for i in range(Np):
            for j in range(i + 1, Np):
                k = (idx(i, t), idx(j, t))
                onehot[k] = onehot.get(k, 0.0) + 2.0
    add_normalized(onehot, lambda_onehot)

    # --- P 嗜好: i@t-1 → j@t に -P[i,j] ---
    pref: Qubo = {}
    for t in range(1, T_steps):
        for i in range(Np):
            for j in range(Np):
                k = (idx(i, t - 1), idx(j, t))
                pref[k] = pref.get(k, 0.0) - P[i, j]
    add_normalized(pref, lambda_P)

    # --- 訪問分散: 各ゾーンへの偏りを抑制 ---
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

    # --- 外部ノード(0)の出入り抑制 ---
    entry: Qubo = {}
    for t in range(T_steps - 1):
        ii, jj = idx(0, t), idx(0, t + 1)
        entry[(ii, ii)] = entry.get((ii, ii), 0.0) + 1.0
        entry[(jj, jj)] = entry.get((jj, jj), 0.0) + 1.0
        entry[(ii, jj)] = entry.get((ii, jj), 0.0) - 2.0
    add_normalized(entry, lambda_entry)

    # --- 内部移動の平滑化 ---
    move: Qubo = {}
    for i in range(1, Np):
        for t in range(T_steps - 1):
            ii, jj = idx(i, t), idx(i, t + 1)
            move[(ii, ii)] = move.get((ii, ii), 0.0) + 1.0
            move[(jj, jj)] = move.get((jj, jj), 0.0) + 1.0
            move[(ii, jj)] = move.get((ii, jj), 0.0) - 2.0
    add_normalized(move, lambda_move)

    return Q

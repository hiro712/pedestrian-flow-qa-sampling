"""
M11(a)+(b): 縮小サブ問題（1時間スライス、2^(N+1)=2048通り）における
有効逆温度 beta_eff の推定と、経験分布とボルツマン分布間の全変動距離 (TV距離)。

時刻 t を1つ固定すると、QUBO のうち時刻 t 内の変数同士のみを結合する項
（one-hot 制約 L_unique の対角ブロック、および L_div/L_entry/L_move の対角線形項）
だけが「局所サブQUBO」 Q_sub を構成する。これは厳密に Np=11 変数 → 2^11=2048
通りなので、全列挙によって厳密なボルツマン分布 P(x) ∝ exp(-beta * E_sub(x))
と分配関数を計算できる（Benedetti et al., arXiv:1510.07611 と同様の
「厳密に扱える部分系で検証する」アプローチ）。

経験分布は、サンプルセットの生のバイナリサンプル（修復前）から時刻 t の
活性化パターンを抜き出して構成する。これにより、サンプラーの出力が
QUBO 自身が定義するボルツマン分布にどれだけ近いかを直接検証できる。

beta_eff は、経験分布と厳密ボルツマン分布間の KL ダイバージェンスを最小化する
beta として最尤推定する。

使い方:
    uv run python analysis/boltzmann_subproblem.py results/sqa_30k --t 6
    uv run python analysis/boltzmann_subproblem.py results/qa_advantage2_30k --t 6
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from src.graph import get_distances
from src.qubo import build_qubo
from src.transition import build_transition_P

LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5


def idx(i: int, t: int, T_steps: int) -> int:
    return i * T_steps + t


def build_local_subqubo(Q: dict, t: int, Np: int, T_steps: int) -> np.ndarray:
    """時刻 t の変数同士のみを結合する項を抽出し、(Np, Np) 行列として返す
    （対角=線形項、非対角=結合項の半分ずつ；energy = sum_i M[i,i] x_i + sum_{i<j} 2 M[i,j] x_i x_j
    となるよう対称行列に整形する）。"""
    M = np.zeros((Np, Np))
    targets = {idx(i, t, T_steps): i for i in range(Np)}
    for (u, v), w in Q.items():
        if u in targets and v in targets:
            iu, iv = targets[u], targets[v]
            if iu == iv:
                M[iu, iu] += w
            else:
                M[iu, iv] += w / 2.0
                M[iv, iu] += w / 2.0
    return M


def enumerate_energies(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全 2^Np 状態のエネルギーを計算する。"""
    Np = M.shape[0]
    states = np.array(list(product([0, 1], repeat=Np)), dtype=float)  # (2^Np, Np)
    # E(x) = x^T M x  (M は対称、対角=線形項として上の構成で正しく計算される)
    E = np.einsum("si,ij,sj->s", states, M, states)
    return states, E


def empirical_distribution(samples_bits: np.ndarray) -> np.ndarray:
    """サンプル群（n_samples, Np)の0/1配列から、2^Np 通りの経験的頻度分布を作る。"""
    Np = samples_bits.shape[1]
    n_states = 2 ** Np
    weights = (1 << np.arange(Np - 1, -1, -1))
    state_ids = samples_bits.astype(int) @ weights
    counts = np.bincount(state_ids, minlength=n_states)
    return counts / counts.sum()


def state_id_for_enumeration(states: np.ndarray) -> np.ndarray:
    Np = states.shape[1]
    weights = (1 << np.arange(Np - 1, -1, -1))
    return states.astype(int) @ weights


def neg_log_likelihood(beta: float, E: np.ndarray, emp_counts: np.ndarray) -> float:
    # log P(x) = -beta E(x) - log Z(beta);  Z(beta) = sum exp(-beta E)
    m = -beta * E
    m_max = m.max()
    log_Z = m_max + np.log(np.sum(np.exp(m - m_max)))
    log_p = m - log_Z
    return -float(np.sum(emp_counts * log_p))


def fit_beta_eff(E: np.ndarray, emp_counts: np.ndarray) -> dict:
    res = minimize_scalar(
        neg_log_likelihood, args=(E, emp_counts),
        bounds=(1e-4, 50.0), method="bounded",
        options={"xatol": 1e-6},
    )
    beta_eff = float(res.x)
    return {"beta_eff": beta_eff, "T_eff": 1.0 / beta_eff, "neg_log_lik": float(res.fun)}


def boltzmann_probs(E: np.ndarray, beta: float) -> np.ndarray:
    m = -beta * E
    m -= m.max()
    p = np.exp(m)
    return p / p.sum()


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def load_full_sampleset(path: Path, Np: int, T_steps: int, t: int) -> np.ndarray:
    """sampleset.json を読み、時刻 t における (n_samples, Np) の0/1配列を返す。"""
    print(f"  loading {path} (this may take a while for large files) ...")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cols = [idx(i, t, T_steps) for i in range(Np)]
    n = len(d["samples"])
    bits = np.zeros((n, Np), dtype=np.int8)
    for row, sample in enumerate(d["samples"]):
        for k, c in enumerate(cols):
            bits[row, k] = sample.get(str(c), sample.get(c, 0))
    return bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, help="例: results/sqa_30k")
    parser.add_argument("--t", type=int, default=6, help="解析対象の時刻スライス (0-indexed)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    label = result_dir.name
    t = args.t

    with open(result_dir / "results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    C_history = np.array(res["C"])
    alpha, beta_param = res["alpha"], res["beta"]
    T_steps, N = C_history.shape
    Np = N + 1
    assert 0 <= t < T_steps

    print(f"[{label}] Reconstructing QUBO (t={t}) ...")
    distances, _ = get_distances()
    P = build_transition_P(C_history, distances, alpha, beta_param)
    Q = build_qubo(P, T_steps, LAMBDA_ONEHOT, LAMBDA_P, LAMBDA_DIV, LAMBDA_ENTRY, LAMBDA_MOVE)

    M = build_local_subqubo(Q, t, Np, T_steps)
    states, E = enumerate_energies(M)
    print(f"[{label}] sub-QUBO enumerated: {len(E)} microstates "
          f"(E range [{E.min():.3f}, {E.max():.3f}])")

    bits = load_full_sampleset(result_dir / "sampleset.json", Np, T_steps, t)
    print(f"[{label}] n_samples = {len(bits)}")

    # 経験分布（厳密列挙の状態順序に揃える）
    state_ids = state_id_for_enumeration(states)
    order = np.argsort(state_ids)
    states_sorted, E_sorted, ids_sorted = states[order], E[order], state_ids[order]

    weights = (1 << np.arange(Np - 1, -1, -1))
    emp_state_ids = bits.astype(int) @ weights
    emp_counts_full = np.bincount(emp_state_ids, minlength=2 ** Np).astype(float)
    emp_counts = emp_counts_full[ids_sorted]
    p_emp = emp_counts / emp_counts.sum()

    fit = fit_beta_eff(E_sorted, emp_counts)
    print(f"[{label}] beta_eff (MLE) = {fit['beta_eff']:.4f}   T_eff = {fit['T_eff']:.4f}")

    p_boltz_fit = boltzmann_probs(E_sorted, fit["beta_eff"])
    p_boltz_unit = boltzmann_probs(E_sorted, 1.0)

    tv_fit = tv_distance(p_emp, p_boltz_fit)
    tv_unit = tv_distance(p_emp, p_boltz_unit)
    print(f"[{label}] TV distance (empirical vs Boltzmann @ beta_eff={fit['beta_eff']:.3f}) = {tv_fit:.4f}")
    print(f"[{label}] TV distance (empirical vs Boltzmann @ beta=1, nominal QUBO units)   = {tv_unit:.4f}")

    # 占有数 k = sum(x) でグルーピングした周辺分布も報告（可視化しやすい）
    k_emp = states_sorted.sum(axis=1).astype(int)
    k_max = Np
    k_dist_emp = np.bincount(np.repeat(k_emp, 1), weights=p_emp, minlength=k_max + 1)
    k_dist_boltz = np.bincount(np.repeat(k_emp, 1), weights=p_boltz_fit, minlength=k_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(np.arange(k_max + 1) - 0.18, k_dist_emp, width=0.36, label="empirical")
    axes[0].bar(np.arange(k_max + 1) + 0.18, k_dist_boltz, width=0.36,
                label=fr"Boltzmann ($\beta_{{\rm eff}}$={fit['beta_eff']:.3f})")
    axes[0].set_xlabel(r"number of active nodes $k=\sum_i x_{i,t}$")
    axes[0].set_ylabel("probability")
    axes[0].set_title(f"{label}: occupancy distribution at t={t}")
    axes[0].legend(fontsize=8)

    order_E = np.argsort(E_sorted)
    axes[1].plot(E_sorted[order_E], p_emp[order_E], ".", ms=3, label="empirical")
    axes[1].plot(E_sorted[order_E], p_boltz_fit[order_E], "-", lw=1.5,
                 label=fr"Boltzmann fit ($\beta_{{\rm eff}}$={fit['beta_eff']:.3f}, TV={tv_fit:.3f})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sub-QUBO energy $E_{sub}(x)$")
    axes[1].set_ylabel("probability (log scale)")
    axes[1].set_title(f"{label}: microstate distribution at t={t}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = result_dir / f"boltzmann_subproblem_t{t}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[{label}] figure saved to {out_path}")

    summary = {
        "label": label,
        "t": t,
        "Np": Np,
        "n_microstates": int(2 ** Np),
        "n_samples": int(len(bits)),
        "beta_eff": fit["beta_eff"],
        "T_eff": fit["T_eff"],
        "tv_distance_at_beta_eff": tv_fit,
        "tv_distance_at_beta_1": tv_unit,
        "violation_rate_at_t": float(np.mean(k_emp_counts := bits.sum(axis=1) != 1)),
    }
    summary_path = result_dir / f"boltzmann_subproblem_t{t}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{label}] summary saved to {summary_path}")


if __name__ == "__main__":
    main()

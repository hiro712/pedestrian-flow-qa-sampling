"""
M11(a)+(b): estimating the effective inverse temperature beta_eff on a
reduced sub-problem (one time slice, 2^(N+1)=2048 states), and the total
variation (TV) distance between the empirical distribution and the
Boltzmann distribution.

Fixing a single time step t, the only QUBO terms that couple variables
within time step t (the diagonal block of the one-hot constraint L_unique,
and the diagonal linear terms of L_div/L_entry/L_move) form a "local
sub-QUBO" Q_sub. This has exactly Np=11 variables -> 2^11=2048 states, so
full enumeration gives the exact Boltzmann distribution
P(x) ∝ exp(-beta * E_sub(x)) and partition function (the same "verify on an
exactly tractable subsystem" approach as Benedetti et al., arXiv:1510.07611).

The empirical distribution is built by extracting the activation pattern at
time t from the sampleset's raw binary samples (before repair). This lets us
directly check how close the sampler's output is to the Boltzmann
distribution defined by the QUBO itself.

beta_eff is the maximum-likelihood estimate of beta that minimizes the KL
divergence between the empirical distribution and the exact Boltzmann
distribution.

Usage:
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
    """Extract the terms coupling only variables at time t and return them as
    an (Np, Np) matrix (diagonal = linear terms, off-diagonal = half the
    coupling term each; shaped into a symmetric matrix so that
    energy = sum_i M[i,i] x_i + sum_{i<j} 2 M[i,j] x_i x_j)."""
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
    """Compute the energy of all 2^Np states."""
    Np = M.shape[0]
    states = np.array(list(product([0, 1], repeat=Np)), dtype=float)  # (2^Np, Np)
    # E(x) = x^T M x  (M is symmetric; diagonal = linear terms, correctly computed by the construction above)
    E = np.einsum("si,ij,sj->s", states, M, states)
    return states, E


def empirical_distribution(samples_bits: np.ndarray) -> np.ndarray:
    """Build the empirical frequency distribution over 2^Np states from a
    (n_samples, Np) array of 0/1 samples."""
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
    """Read sampleset.json and return the (n_samples, Np) 0/1 array at time t."""
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
    parser.add_argument("result_dir", type=str, help="e.g. results/sqa_30k")
    parser.add_argument("--t", type=int, default=6, help="time slice to analyze (0-indexed)")
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

    # Empirical distribution (aligned to the exact-enumeration state order)
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

    # Also report the marginal distribution grouped by occupancy count k = sum(x) (easier to visualize)
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

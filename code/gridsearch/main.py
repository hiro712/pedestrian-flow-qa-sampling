# filename: grid_search_alpha_beta.py
from typing import Optional, Tuple, Dict, Any, Callable, List
import numpy as np
import openjij as oj
import dimod
import csv
import json
import os
from datetime import datetime

from solve.main import (
    QASolver,
    SASolver,
    SQASolver,
)  # samplers (example alternatives left in comments below)


# =========================
# Utilities
# =========================
def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize (rows whose sum is 0 stay 0)."""
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(m, np.where(rowsum > eps, rowsum, 1.0))
        out = np.where(rowsum > eps, out, 0.0)
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_progress_cb(info: Dict[str, Any]) -> None:
    msg = (
        f"[{info.get('when')}] {info.get('stage','?')} "
        f"{info.get('step','')}/{info.get('total','')} "
        f"α={info.get('alpha','-')}, β={info.get('beta','-')}, "
        f"loss={info.get('loss','-')}, rmse={info.get('rmse','-')}, "
        f"best=({info.get('best_alpha','-')},{info.get('best_beta','-')} "
        f"L={info.get('best_loss','-')},R={info.get('best_rmse','-')}) "
        f"viol={info.get('violation_rate','-')}"
    )
    print(msg)


def _log_progress_files(
    info: Dict[str, Any],
    history_csv: str = "history.csv",
) -> None:
    # CSV output only
    fields = [
        "when",
        "stage",
        "step",
        "total",
        "alpha",
        "beta",
        "loss",
        "rmse",
        "best_alpha",
        "best_beta",
        "best_loss",
        "best_rmse",
        "violation_rate",
    ]
    os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
    new_file = not os.path.exists(history_csv)
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        row = {k: info.get(k, "") for k in fields}
        w.writerow(row)


def _load_history_cache(
    history_csv: str, stage_name: str
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """
    Load existing data from history.csv and return a dict mapping
    (alpha, beta) -> {loss, rmse, violation_rate}.
    """
    cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
    if not os.path.exists(history_csv):
        return cache

    try:
        with open(history_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") == stage_name:
                    try:
                        alpha = float(row["alpha"])
                        beta = float(row["beta"])
                        loss = float(row["loss"])
                        rmse = float(row["rmse"])
                        violation_rate = float(row.get("violation_rate", 0.0))
                        cache[(alpha, beta)] = {
                            "loss": loss,
                            "rmse": rmse,
                            "violation_rate": violation_rate,
                        }
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print(f"Warning: Failed to load history cache: {e}")

    return cache


# =========================
# Transition matrix builder (assumes distances are already normalized)
# =========================
def _build_transition_P(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) normalized
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Build weights W from the most recent proportions p_prev, distance, and
    popularity, then row-normalize to return P (outside = 0).
    """
    T, N = C_history.shape
    Np = N + 1
    if distances.shape != (Np, Np):
        raise ValueError(f"distances shape must be {(Np, Np)}, got {distances.shape}")

    p_hist = _safe_row_normalize(C_history)

    prev_sum = C_history[-2].sum() if T >= 2 else C_history[-1].sum()
    cur_sum = C_history[-1].sum()
    denom = prev_sum if prev_sum > 0 else 1.0
    delta_plus = max(cur_sum - prev_sum, 0.0) / denom
    prev_idx = -2 if T >= 2 else -1
    p_prev = np.concatenate(([delta_plus], p_hist[prev_idx] * (1.0 - delta_plus)))

    # Apply alpha/beta
    W = np.exp(-alpha * distances) * (p_prev[None, :] ** beta)
    P = _safe_row_normalize(W)
    return P


# =========================
# Build the QUBO
# =========================
def _build_qubo_from_P(
    P: np.ndarray,
    T_steps: int,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
) -> Tuple[Dict[Tuple[int, int], float], int]:
    """
    Build a (normalized) QUBO from P. Returns: (Q, Np)
    """
    Np = P.shape[0]
    Q: Dict[Tuple[int, int], float] = {}

    def idx(i: int, t: int) -> int:
        return i * T_steps + t

    def apply_normalized(base_dict: dict, lam: float):
        if not base_dict:
            return
        max_abs = max(abs(v) for v in base_dict.values()) or 1.0
        for key, raw_val in base_dict.items():
            Q[key] = Q.get(key, 0.0) + lam * (raw_val / max_abs)

    # One-hot (penalize multiple selections at the same time step; 0 or 1 is allowed)
    base_onehot: Dict[Tuple[int, int], float] = {}
    for t in range(T_steps):
        for i in range(Np):
            for j in range(i + 1, Np):
                base_onehot[(idx(i, t), idx(j, t))] = (
                    base_onehot.get((idx(i, t), idx(j, t)), 0.0) + 2.0
                )
    apply_normalized(base_onehot, lambda_onehot)

    # P preference (-P[i,j] for i at t-1 -> j at t)
    base_P: Dict[Tuple[int, int], float] = {}
    for t in range(1, T_steps):
        for i in range(Np):
            for j in range(Np):
                base_P[(idx(i, t - 1), idx(j, t))] = (
                    base_P.get((idx(i, t - 1), idx(j, t)), 0.0) - P[i, j]
                )
    apply_normalized(base_P, lambda_P)

    # Dispersion (avoid bias toward the same area)
    avg_visits = T_steps / Np
    base_div: Dict[Tuple[int, int], float] = {}
    for i in range(Np):
        for t in range(T_steps):
            ii = idx(i, t)
            base_div[(ii, ii)] = base_div.get((ii, ii), 0.0) + (1 - 2 * avg_visits)
            for t2 in range(t + 1, T_steps):
                jj = idx(i, t2)
                base_div[(ii, jj)] = base_div.get((ii, jj), 0.0) + 2.0
    apply_normalized(base_div, lambda_div)

    # Suppress toggling in/out of the outside node (0)
    base_entry: Dict[Tuple[int, int], float] = {}
    for t in range(T_steps - 1):
        ii = idx(0, t)
        jj = idx(0, t + 1)
        base_entry[(ii, ii)] = base_entry.get((ii, ii), 0.0) + 1.0
        base_entry[(jj, jj)] = base_entry.get((jj, jj), 0.0) + 1.0
        base_entry[(ii, jj)] = base_entry.get((ii, jj), 0.0) - 2.0
    apply_normalized(base_entry, lambda_entry)

    # Smoothing of internal moves
    base_move: Dict[Tuple[int, int], float] = {}
    for i in range(1, Np):
        for t in range(T_steps - 1):
            ii = idx(i, t)
            jj = idx(i, t + 1)
            base_move[(ii, ii)] = base_move.get((ii, ii), 0.0) + 1.0
            base_move[(jj, jj)] = base_move.get((jj, jj), 0.0) + 1.0
            base_move[(ii, jj)] = base_move.get((ii, jj), 0.0) - 2.0
    apply_normalized(base_move, lambda_move)

    return Q, Np


# =========================
# Sample repair / decoding
# =========================
def _fix_sample_onehot(
    sample: Dict[int, int],
    Np: int,
    T_steps: int,
    rng: np.random.Generator,
) -> Dict[int, int]:
    """
    For each one-hot violation (2 or more 1s at the same time step), keep one
    at random and set the rest to 0. Leaves 0 or 1 active unchanged (not a
    violation).
    """
    fixed = dict(sample)
    for t in range(T_steps):
        actives = [i for i in range(Np) if sample.get(i * T_steps + t, 0) == 1]
        if len(actives) >= 2:
            keep = int(rng.choice(actives))
            for i in actives:
                fixed[i * T_steps + t] = 1 if i == keep else 0
    return fixed


def _decode_traj_from_sample(
    sample: Dict[int, int], Np: int, T_steps: int
) -> List[int]:
    """
    Convert a sample (assumed to satisfy one-hot) into a trajectory list.
    0 active -> outside (0).
    """
    traj = []
    for t in range(T_steps):
        actives = [i for i in range(Np) if sample.get(i * T_steps + t, 0) == 1]
        if len(actives) == 1:
            u = actives[0]
        elif len(actives) == 0:
            u = 0
        else:
            # Should not normally reach here (already resolved by _fix_sample_onehot)
            u = actives[0]
        traj.append(u)
    return traj


def _reconstruct_proportions_from_trajs(
    traj_list: List[List[int]],
    T_steps: int,
    N: int,  # number of internal areas (excluding outside)
) -> np.ndarray:
    """
    Reconstruct the internal-area proportions p' (T x N) at each time step
    from the sampled trajectories.
    """
    num_samples = len(traj_list)
    if num_samples == 0:
        return np.zeros((T_steps, N), dtype=float)

    counts = np.zeros((T_steps, N), dtype=float)
    for traj in traj_list:
        for t in range(T_steps):
            u = traj[t]
            if 1 <= u <= N:
                counts[t, u - 1] += 1.0  # internal zones 1..N -> 0..N-1

    p_prime = counts / float(num_samples)
    return p_prime


# =========================
# Grid search core (outputs only history.csv)
# =========================
def grid_search_alpha_beta(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) normalized
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    alphas: List[float],
    betas: List[float],
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    history_csv: str = "history.csv",
    stage_name: str = "grid",
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    Minimize the discrepancy (LOSS/RMSE) between p and p' over a grid of
    (alpha, beta). Outputs only history.csv (no energy plots etc.).
    For the RMSE computation, constraint-violating samples are randomly
    repaired to one-hot before computing p'.

    Returns: (alpha*, beta*, best_loss, best_rmse, meta={'history':[...]})
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")

    # Precompute
    T, N = C_history.shape
    T_steps = T
    p_true = _safe_row_normalize(C_history)  # row-normalized (proportions)

    # Load cache (skip existing results for the same stage)
    cached = _load_history_cache(history_csv, stage_name)

    total = len(alphas) * len(betas)
    step = 0
    best = {"alpha": None, "beta": None, "loss": float("inf"), "rmse": float("inf")}
    history: List[Dict[str, Any]] = []

    # Restore the best result from cache
    for (a, b), result in cached.items():
        loss = result["loss"]
        rmse = result["rmse"]
        viol = result["violation_rate"]

        history.append(
            {
                "alpha": float(a),
                "beta": float(b),
                "loss": loss,
                "rmse": rmse,
                "onehot_violation_rate": float(viol),
            }
        )
        if loss < best["loss"]:
            best.update(
                {"alpha": float(a), "beta": float(b), "loss": loss, "rmse": rmse}
            )

    # RNG for random tie-breaking (reproducible via seed)
    rng = np.random.default_rng(seed)

    for a in alphas:
        for b in betas:
            step += 1

            # Skip if already evaluated for the same stage_name
            if (a, b) in cached:
                progress_cb(
                    {
                        "when": _now(),
                        "stage": stage_name,
                        "step": step,
                        "total": total,
                        "alpha": float(a),
                        "beta": float(b),
                        "loss": cached[(a, b)]["loss"],
                        "rmse": cached[(a, b)]["rmse"],
                        "best_alpha": best["alpha"],
                        "best_beta": best["beta"],
                        "best_loss": best["loss"],
                        "best_rmse": best["rmse"],
                        "violation_rate": cached[(a, b)]["violation_rate"],
                    }
                )
                continue

            # 1) Build P
            P = _build_transition_P(C_history, distances, a, b)

            # 2) Build the QUBO
            Q, Np = _build_qubo_from_P(
                P,
                T_steps,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
            )

            # 3) Sampling
            # bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            # sampler = oj.SASampler()
            # sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed, num_sweeps=100)

            solver = SASolver()
            sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
            if seed is not None:
                sample_config["seed"] = seed
            sampleset = solver.solve(Q, sample_config=sample_config)

            # solver = SQASolver()
            # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
            # if seed is not None:
            #     sample_config["seed"] = seed
            # sampleset = solver.solve(Q, sample_config=sample_config)

            # solver = QASolver()
            # sample_config: Dict[str, Any] = {
            #     "num_reads": num_reads,
            #     "annealing_time": 20,
            # }
            # if seed is not None:
            #     sample_config["seed"] = seed
            # sampleset = solver.solve(Q, sample_config=sample_config)

            # 4) Violation rate (judged from raw samples: 0/1 is OK, 2+ is a violation)
            violations = 0
            total_slots = 0
            for sample in sampleset.samples():
                for t in range(T_steps):
                    total_slots += 1
                    cnt = sum(sample.get(i * T_steps + t, 0) for i in range(Np))
                    if cnt not in (0, 1):
                        violations += 1
            violation_rate = violations / total_slots if total_slots else 0.0

            # 5) Repair to satisfy constraints -> decode to trajectories -> reconstruct p'
            traj_list: List[List[int]] = []
            for sample in sampleset.samples():
                fixed = _fix_sample_onehot(dict(sample), Np, T_steps, rng)
                traj = _decode_traj_from_sample(fixed, Np, T_steps)
                traj_list.append(traj)

            p_prime = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)

            # 6) Loss (sum of squares of p vs p') & RMSE
            diff = p_true - p_prime
            loss = float(np.sum(diff**2))
            rmse = float(np.sqrt(loss / diff.size))

            # 7) Progress notification & append to history.csv
            info = {
                "when": _now(),
                "stage": stage_name,
                "step": step,
                "total": total,
                "alpha": float(a),
                "beta": float(b),
                "loss": round(loss, 6),
                "rmse": round(rmse, 6),
                "violation_rate": round(violation_rate, 6),
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_loss": best["loss"],
                "best_rmse": best["rmse"],
            }
            progress_cb(info)
            _log_progress_files(info, history_csv=history_csv)

            history.append(
                {
                    "alpha": float(a),
                    "beta": float(b),
                    "loss": loss,
                    "rmse": rmse,
                    "onehot_violation_rate": float(violation_rate),
                }
            )
            if loss < best["loss"]:
                best.update(
                    {"alpha": float(a), "beta": float(b), "loss": loss, "rmse": rmse}
                )

    meta = {"history": history}
    return best["alpha"], best["beta"], best["loss"], best["rmse"], meta

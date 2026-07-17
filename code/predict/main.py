from typing import Optional, Tuple, Dict, Any, Callable, List
import numpy as np
import openjij as oj
import dimod
import csv
import matplotlib
import json
import hashlib
import os
from datetime import datetime

from solve.main import QASolver, SASolver, SQASolver

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================
def _arr_hash(a: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(a.tobytes())
    return h.hexdigest()


def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize (rows whose sum is 0 stay 0)."""
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(m, np.where(rowsum > eps, rowsum, 1.0))
        out = np.where(rowsum > eps, out, 0.0)
    return out


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


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
    log_txt: str = "progress.log",
    history_csv: str = "history.csv",
) -> None:
    _ensure_parent_dir(log_txt)
    _ensure_parent_dir(history_csv)
    # JSON Lines
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    # CSV
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
    new_file = not os.path.exists(history_csv)
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        row = {k: info.get(k, "") for k in fields}
        w.writerow(row)


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
# QUBO & sampling (trajectory length is T = C_history.shape[0])
# =========================
def _sample_trajectories(
    P: np.ndarray,
    T_steps: int,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
) -> Tuple[List[List[int]], float]:
    """
    Build the QUBO and sample with OpenJij. Returns trajectories of length
    T_steps and the one-hot violation rate.
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

    # Sampling
    # bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    # sampler = oj.SASampler()
    # sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed, num_sweeps=100)

    # solver = SASolver()
    # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
    # if seed is not None:
    #     sample_config["seed"] = seed
    # sampleset = solver.solve(Q, sample_config=sample_config)

    # solver = SQASolver()
    # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
    # if seed is not None:
    #     sample_config["seed"] = seed
    # sampleset = solver.solve(Q, sample_config=sample_config)

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")  # load environment variables from .env.local
    print("DWAVE_SOLVER_NAME:", os.getenv("DWAVE_SOLVER_NAME"))
    solver = QASolver()
    sample_config: Dict[str, Any] = {
        "num_reads": num_reads,
        "annealing_time": 20,
    }
    if seed is not None:
        sample_config["seed"] = seed
    sampleset = solver.solve(Q, sample_config=sample_config)

    # ---------- Added below: helpers for random decoding / one-hot repair / energy recomputation ----------
    # RNG for random tie-breaking (reproducible via seed)
    rng = np.random.default_rng(seed)

    def _decode_traj(sample: Dict[int, int]) -> List[int]:
        """Random choice if multiple 1s at the same time step; outside (0) if none."""
        traj = []
        for t in range(T_steps):
            actives = [i for i in range(Np) if sample.get(idx(i, t), 0) == 1]
            if len(actives) == 1:
                u = actives[0]
            elif len(actives) == 0:
                u = 0
            else:
                u = int(rng.choice(actives))
            traj.append(u)
        return traj

    def _fix_sample_onehot(sample: Dict[int, int]) -> Dict[int, int]:
        """For each one-hot violation (2 or more 1s at the same time step),
        keep one at random and set the rest to 0. Leaves 0 or 1 active
        unchanged (not a violation)."""
        fixed = dict(sample)
        for t in range(T_steps):
            actives = [i for i in range(Np) if sample.get(idx(i, t), 0) == 1]
            if len(actives) >= 2:
                keep = int(rng.choice(actives))
                for i in actives:
                    fixed[idx(i, t)] = 1 if i == keep else 0
        return fixed

    def _qubo_energy(sample: Dict[int, int]) -> float:
        """Recompute the energy against Q (the normalized QUBO)."""
        e = 0.0
        for (u, v), w in Q.items():
            e += w * sample.get(u, 0) * sample.get(v, 0)
        return float(e)

    # ---------- End of added section -----------------------------------------------------

    # Save the sampleset as JSON (to_serializable() saves all data)
    sampleset_data = sampleset.to_serializable()
    with open("sampleset.json", "w", encoding="utf-8") as f:
        json.dump(sampleset_data, f, ensure_ascii=False, indent=2)

    # Save the trajectory CSV (random tie-break; empty -> outside 0)
    with open("traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for sample in sampleset.samples():
            traj = _decode_traj(dict(sample))
            writer.writerow(traj)

    # Energy distribution
    # (A) energy reported by the QA hardware (for reference); also saved under the legacy filename for compatibility
    hw_energies = sampleset.record.energy
    plt.figure()
    plt.hist(hw_energies, bins=50)
    plt.xlabel("Energy (hardware reported)")
    plt.ylabel("Frequency")
    plt.title("Sampling Energy Histogram (hardware)")
    plt.savefig("energy_histogram_hw.png")
    plt.savefig("energy_histogram.png")  # kept for compatibility: legacy filename
    plt.close()

    # (B) recompute and overlay "raw sample" and "one-hot-repaired sample" energies against the normalized QUBO Q
    raw_energies_recalc = np.array(
        [_qubo_energy(dict(sample)) for sample in sampleset.samples()], dtype=float
    )
    fixed_energies_recalc = np.array(
        [
            _qubo_energy(_fix_sample_onehot(dict(sample)))
            for sample in sampleset.samples()
        ],
        dtype=float,
    )

    all_vals = np.concatenate([raw_energies_recalc, fixed_energies_recalc])
    # Also handle the case where all values are identical
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    bins_arg: Any
    if np.isclose(vmin, vmax):
        bins_arg = 50
    else:
        bins_arg = np.linspace(vmin, vmax, 50).tolist()

    plt.figure()
    plt.hist(raw_energies_recalc, bins=bins_arg, alpha=0.6, label="Raw (recomputed)")
    plt.hist(
        fixed_energies_recalc,
        bins=bins_arg,
        alpha=0.6,
        label="Fixed (random tie-break; recomputed)",
    )
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Raw vs Fixed (recomputed)")
    plt.legend()
    plt.savefig("energy_histogram_overlay.png")
    plt.close()

    # (C) histogram of Fixed only
    plt.figure()
    plt.hist(fixed_energies_recalc, bins=50, alpha=0.8)
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Fixed (random tie-break; recomputed)")
    plt.savefig("energy_histogram_fixed.png")
    plt.close()

    # One-hot violation rate (0 or 1 is OK, 2+ is a violation)
    violations = 0
    total_slots = 0
    for sample in sampleset.samples():
        for t in range(T_steps):
            total_slots += 1
            cnt = sum(sample.get(idx(i, t), 0) for i in range(Np))
            if cnt not in (0, 1):
                violations += 1
    violation_rate = violations / total_slots if total_slots else 0.0

    # Trajectory list (same random decoding as the CSV)
    traj_list: List[List[int]] = []
    for sample in sampleset.samples():
        traj_list.append(_decode_traj(dict(sample)))

    return traj_list, violation_rate


# =========================
# Proportions p' reconstruction from trajectories
# =========================
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
# Grid runner (minimizes p vs p')
# =========================
def _load_history_cache(
    history_csv: str, stage_name: str
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """
    Load existing data from history.csv and return a dict mapping
    (alpha, beta) -> {loss, rmse, ...}.
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


def _run_grid(
    stage_name: str,
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
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    Minimize the sum-of-squares LOSS and RMSE between p and p' (row-normalized
    proportions) over the given grid. Skips entries already present in
    history.csv.
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    T, N = C_history.shape
    T_steps = T
    p_true = _safe_row_normalize(C_history)  # row-normalized (proportions)

    # Load existing data
    cached_results = _load_history_cache(history_csv, stage_name)
    print(
        f"[{stage_name}] Loaded {len(cached_results)} cached results from {history_csv}"
    )

    total = len(alphas) * len(betas)
    step = 0
    best = {"alpha": None, "beta": None, "loss": float("inf"), "rmse": float("inf")}
    history = []

    # Restore the best value from cache
    for (a, b), result in cached_results.items():
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

    for a in alphas:
        for b in betas:
            step += 1

            # Skip if present in cache
            if (a, b) in cached_results:
                result = cached_results[(a, b)]
                print(f"[{stage_name}] Skipping {step}/{total} α={a}, β={b} (cached)")
                continue
            # 1) P
            P = _build_transition_P(C_history, distances, a, b)
            # 2) Sample -> trajectories
            traj_list, viol = _sample_trajectories(
                P,
                T_steps,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
                seed,
                num_reads,
            )
            # 3) Reconstruct p'
            p_prime = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
            # 4) Loss (sum of squares of p vs p') & RMSE
            diff = p_true - p_prime
            loss = float(np.sum(diff**2))
            rmse = float(np.sqrt(loss / diff.size))

            # Progress notification & logging
            info = {
                "when": _now(),
                "stage": stage_name,
                "step": step,
                "total": total,
                "alpha": float(a),
                "beta": float(b),
                "loss": round(loss, 6),
                "rmse": round(rmse, 6),
                "violation_rate": round(viol, 6),
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_loss": best["loss"],
                "best_rmse": best["rmse"],
            }
            progress_cb(info)
            _log_progress_files(info)

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

    return best["alpha"], best["beta"], best["loss"], best["rmse"], {"history": history}


# =========================
# Two-stage search: Stage1 (wide and coarse) -> Stage2 (fine-grained locally)
# =========================
def optimize_alpha_beta_two_stage(
    C_history: np.ndarray,
    distances: np.ndarray,  # assumed already normalized
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    # Stage1 (wide and coarse)
    alphas_stage1: Optional[List[float]] = None,
    betas_stage1: Optional[List[float]] = None,
    # Stage2 (fine-grained locally): +- window and step
    stage2_alpha_window: float = 0.6,
    stage2_alpha_step: float = 0.1,
    stage2_beta_window: float = 0.6,
    stage2_beta_step: float = 0.05,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cache_path: str = "alpha_beta_cache.json",
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    Optimize (alpha, beta) via an automatic two-stage search (minimizing
    LOSS/RMSE between p and p').
    Returns: (alpha*, beta*, best_loss, best_rmse, meta)
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    # Stage1 defaults (wide and coarse) -- cast numpy.float64 -> float for type consistency
    if alphas_stage1 is None:
        alphas_stage1 = list(map(float, np.arange(0.0, 6.0 + 1e-9, 0.5)))
    if betas_stage1 is None:
        betas_stage1 = list(map(float, np.arange(0.3, 2.5 + 1e-9, 0.2)))

    # --- Stage 1 ---
    a1, b1, loss1, rmse1, meta1 = _run_grid(
        stage_name="stage1",
        C_history=C_history,
        distances=distances,
        lambda_onehot=lambda_onehot,
        lambda_P=lambda_P,
        lambda_div=lambda_div,
        lambda_entry=lambda_entry,
        lambda_move=lambda_move,
        seed=seed,
        num_reads=num_reads,
        alphas=alphas_stage1,
        betas=betas_stage1,
        progress_cb=progress_cb,
        history_csv="history.csv",
    )

    # Stage2 grid (fine-grained around the best point)
    a_lo = max(0.0, a1 - stage2_alpha_window)
    a_hi = a1 + stage2_alpha_window
    b_lo = max(0.1, b1 - stage2_beta_window)
    b_hi = b1 + stage2_beta_window
    alphas_stage2 = list(map(float, np.arange(a_lo, a_hi + 1e-9, stage2_alpha_step)))
    betas_stage2 = list(map(float, np.arange(b_lo, b_hi + 1e-9, stage2_beta_step)))

    # --- Stage 2 ---
    a2, b2, loss2, rmse2, meta2 = _run_grid(
        stage_name="stage2",
        C_history=C_history,
        distances=distances,
        lambda_onehot=lambda_onehot,
        lambda_P=lambda_P,
        lambda_div=lambda_div,
        lambda_entry=lambda_entry,
        lambda_move=lambda_move,
        seed=seed,
        num_reads=num_reads,
        alphas=alphas_stage2,
        betas=betas_stage2,
        progress_cb=progress_cb,
        history_csv="history.csv",
    )

    # Save to cache (keyed by C, D, and the search conditions)
    key = {
        "C_hash": _arr_hash(C_history),
        "D_hash": _arr_hash(distances),
        "num_reads": num_reads,
        "seed": seed,
        "stage2_window": {
            "alpha": stage2_alpha_window,
            "beta": stage2_beta_window,
            "alpha_step": stage2_alpha_step,
            "beta_step": stage2_beta_step,
        },
    }
    key_str = json.dumps(key, sort_keys=True)

    cache: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    best = {"alpha": a2, "beta": b2, "loss": loss2, "rmse": rmse2}
    cache[key_str] = {
        "key": key,
        "best": best,
        "stage1": {
            "alphas": alphas_stage1,
            "betas": betas_stage1,
            "best_alpha": a1,
            "best_beta": b1,
            "best_loss": loss1,
            "best_rmse": rmse1,
            "history": meta1.get("history", []),
        },
        "stage2": {
            "alphas": alphas_stage2,
            "betas": betas_stage2,
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
            "history": meta2.get("history", []),
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Completion notification
    progress_cb(
        {
            "when": _now(),
            "stage": "done-two-stage",
            "step": "-",
            "total": "-",
            "alpha": a2,
            "beta": b2,
            "loss": loss2,
            "rmse": rmse2,
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
            "violation_rate": "-",
        }
    )

    meta = {
        "stage1": {
            "best_alpha": a1,
            "best_beta": b1,
            "best_loss": loss1,
            "best_rmse": rmse1,
        },
        "stage2": {
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
        },
    }
    return a2, b2, loss2, rmse2, meta


# =========================
# Main prediction (alpha/beta unspecified -> automatic two-stage search)
# =========================
def predict_flow_percent(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) normalized
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    # alpha, beta can be specified (auto two-stage search if unspecified)
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    # Stage1/Stage2 parameters (override if needed)
    alphas_stage1: Optional[List[float]] = None,
    betas_stage1: Optional[List[float]] = None,
    stage2_alpha_window: float = 0.6,
    stage2_alpha_step: float = 0.1,
    stage2_beta_window: float = 0.6,
    stage2_beta_step: float = 0.05,
    # Progress callback
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    # Cache file
    cache_path: str = "alpha_beta_cache.json",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], Dict[str, Any]]:
    """
    Sample individual trajectories via quantum annealing and return P, F, p'.
    When alpha, beta are unspecified, minimizes the discrepancy between p and
    p' via an automatic "wide and coarse -> fine-grained locally" two-stage
    search.
    Returns: (P, F, p_prime, (alpha,beta), meta)
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    T, N = C_history.shape
    T_steps = T

    # Always precompute the row-normalized true proportions p_true
    p_true = _safe_row_normalize(C_history)

    best_loss = None
    best_rmse = None

    if alpha is None or beta is None:
        alpha, beta, best_loss, best_rmse, two_stage_meta = (
            optimize_alpha_beta_two_stage(
                C_history,
                distances,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
                seed,
                num_reads,
                alphas_stage1=alphas_stage1,
                betas_stage1=betas_stage1,
                stage2_alpha_window=stage2_alpha_window,
                stage2_alpha_step=stage2_alpha_step,
                stage2_beta_window=stage2_beta_window,
                stage2_beta_step=stage2_beta_step,
                progress_cb=progress_cb,
                cache_path=cache_path,
            )
        )
        progress_cb(
            {
                "when": _now(),
                "stage": "select",
                "step": "-",
                "total": "-",
                "alpha": alpha,
                "beta": beta,
                "loss": best_loss,
                "rmse": best_rmse,
                "best_alpha": alpha,
                "best_beta": beta,
                "best_loss": best_loss,
                "best_rmse": best_rmse,
                "violation_rate": "-",
            }
        )
    else:
        two_stage_meta = {}

    # Final P
    P = _build_transition_P(C_history, distances, alpha, beta)

    # Sampling -> F and p'
    traj_list, violation_rate = _sample_trajectories(
        P,
        T_steps,
        lambda_onehot,
        lambda_P,
        lambda_div,
        lambda_entry,
        lambda_move,
        seed,
        num_reads,
    )
    # Reconstruct p' and compute the loss
    p_prime_tmp = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
    diff_tmp = p_true - p_prime_tmp
    loss_final = float(np.sum(diff_tmp**2))
    rmse_final = float(np.sqrt(loss_final / diff_tmp.size))

    progress_cb(
        {
            "when": _now(),
            "stage": "sample_final",
            "step": "-",
            "total": "-",
            "alpha": alpha,
            "beta": beta,
            "loss": round(loss_final, 6),
            "rmse": round(rmse_final, 6),
            "best_alpha": alpha,
            "best_beta": beta,
            "best_loss": best_loss if best_loss is not None else round(loss_final, 6),
            "best_rmse": best_rmse if best_rmse is not None else round(rmse_final, 6),
            "violation_rate": violation_rate,
        }
    )

    # Flow F (row-normalized counts of u->v transitions)
    Np = P.shape[0]
    F_counts = np.zeros((Np, Np))
    for traj in traj_list:
        for t in range(T_steps - 1):
            u = traj[t]
            v = traj[t + 1]
            F_counts[u, v] += 1
    F = _safe_row_normalize(F_counts)

    # Reconstruct p' (proportions)
    # Reuse p_prime_tmp computed earlier, if available
    p_prime = (
        _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
        if "p_prime_tmp" not in locals()
        else p_prime_tmp
    )

    # Save
    results = {
        "C": C_history.tolist(),
        "p_true": _safe_row_normalize(C_history).tolist(),
        "p_prime": p_prime.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "alpha": alpha,
        "beta": beta,
        # best_* is the best value found by the search. If not searched (user-specified), use this run's value
        "best_loss": (
            best_loss if best_loss is not None else float(round(loss_final, 6))
        ),
        "best_rmse": (
            best_rmse if best_rmse is not None else float(round(rmse_final, 6))
        ),
        "loss": float(round(loss_final, 6)),
        "rmse": float(round(rmse_final, 6)),
        "meta": two_stage_meta,
    }
    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=4)

    meta = {
        "violation_rate": violation_rate,
        "best_loss": best_loss,
        "best_rmse": best_rmse,
        **two_stage_meta,
    }
    return P, F, p_prime, (alpha, beta), meta

"""
M5 review response: three analyses addressing circularity in the validation.

(a) Leave-One-Out Cross Validation:
    Select (alpha, beta) using the loss with one time block t held out, then
    measure out-of-sample RMSE on the held-out time block t. This removes the
    circularity of "selecting and reporting with the same metric."

(b) Sensitivity analysis of the flow matrix F to +-10% perturbations of
    (alpha, beta).

(c) Bootstrap confidence intervals for RMSE (resampled from the existing
    SQA and QA trajectories).

For fast hyperparameter search, (a) and (b) use the SA solver (the same
default solver as the manuscript's gridsearch.py) with a reduced read count.

Usage:
    uv run python experiments/m5_validation.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.flow import build_flow_matrix, reconstruct_proportions
from src.graph import get_distances
from src.metrics import rmse, squared_loss
from src.qubo import build_qubo
from src.solvers.sa import SASolver
from src.trajectory import decode_traj
from src.transition import _safe_row_normalize, build_transition_P

LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
NUM_SWEEPS = 100

SEARCH_READS = 3000     # for grid search (reduced for speed)
EVAL_READS = 10000      # for final evaluation
SEED = 0

ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
BETA_GRID = [0.35, 0.45, 0.55, 0.65, 0.75]

OUT_DIR = Path("results/m5_validation")


def run_once(C, distances, alpha, beta, num_reads, seed, solver):
    T, N = C.shape
    Np = N + 1
    rng = np.random.default_rng(seed)
    P = build_transition_P(C, distances, alpha, beta)
    Q = build_qubo(P, T, LAMBDA_ONEHOT, LAMBDA_P, LAMBDA_DIV, LAMBDA_ENTRY, LAMBDA_MOVE)
    ss = solver.solve(Q, sample_config={"num_reads": num_reads, "num_sweeps": NUM_SWEEPS, "seed": seed})
    traj = [decode_traj(dict(s), T, Np, rng) for s in ss.samples()]
    p_prime = reconstruct_proportions(traj, T, N)
    F = build_flow_matrix(traj, T, Np)
    return p_prime, F


# ----------------------------------------------------------------------
# (a) Leave-One-Out Cross Validation
# ----------------------------------------------------------------------
def loo_cv(C, distances, solver):
    T, N = C.shape
    p_true = _safe_row_normalize(C)
    fold_results = []

    for t_held in range(T):
        train_mask = np.array([t != t_held for t in range(T)])
        best = {"loss": np.inf, "alpha": None, "beta": None, "p_prime": None}

        for a, b in product(ALPHA_GRID, BETA_GRID):
            seed = SEED + hash((t_held, round(a, 3), round(b, 3))) % 10_000
            p_prime, _ = run_once(C, distances, a, b, SEARCH_READS, seed, solver)
            train_loss = float(np.sum((p_true[train_mask] - p_prime[train_mask]) ** 2))
            if train_loss < best["loss"]:
                best = {"loss": train_loss, "alpha": a, "beta": b, "p_prime": p_prime}

        # Final evaluation: measure held-out RMSE with the selected (alpha, beta) (more reads for evaluation)
        seed_eval = SEED + 777 + t_held
        p_prime_eval, _ = run_once(C, distances, best["alpha"], best["beta"], EVAL_READS, seed_eval, solver)
        held_out_rmse = float(np.sqrt(np.mean((p_true[t_held] - p_prime_eval[t_held]) ** 2)))

        fold_results.append({
            "t_held": t_held,
            "alpha": best["alpha"],
            "beta": best["beta"],
            "held_out_rmse": held_out_rmse,
        })
        print(f"  [LOO fold t={t_held:2d}] selected (alpha={best['alpha']:.2f}, beta={best['beta']:.2f})  "
              f"held-out RMSE = {held_out_rmse:.6f}")

    rmses = np.array([r["held_out_rmse"] for r in fold_results])
    summary = {
        "folds": fold_results,
        "loo_rmse_mean": float(rmses.mean()),
        "loo_rmse_std": float(rmses.std(ddof=1)),
    }
    print(f"[LOO-CV] mean RMSE = {summary['loo_rmse_mean']:.6f} +/- {summary['loo_rmse_std']:.6f}")
    return summary


# ----------------------------------------------------------------------
# (b) sensitivity analysis: +-10% perturbation of (alpha, beta)
# ----------------------------------------------------------------------
def sensitivity_analysis(C, distances, solver, alpha0=0.30, beta0=0.55):
    _, F0 = run_once(C, distances, alpha0, beta0, EVAL_READS, SEED + 1000, solver)

    perturbations = [
        ("alpha-10%", alpha0 * 0.9, beta0),
        ("alpha+10%", alpha0 * 1.1, beta0),
        ("beta-10%", alpha0, beta0 * 0.9),
        ("beta+10%", alpha0, beta0 * 1.1),
    ]
    results = []
    for name, a, b in perturbations:
        _, F = run_once(C, distances, a, b, EVAL_READS, SEED + 1000, solver)
        max_abs_dF = float(np.max(np.abs(F - F0)))
        mean_abs_dF = float(np.mean(np.abs(F - F0)))
        results.append({"perturbation": name, "alpha": a, "beta": b,
                        "max_abs_dF": max_abs_dF, "mean_abs_dF": mean_abs_dF})
        print(f"  [{name}] (alpha={a:.3f}, beta={b:.3f}): "
              f"max|dF|={max_abs_dF:.4f}  mean|dF|={mean_abs_dF:.4f}")
    return {"baseline": {"alpha": alpha0, "beta": beta0}, "perturbations": results}


# ----------------------------------------------------------------------
# (c) bootstrap CI for RMSE (resample existing trajectories)
# ----------------------------------------------------------------------
def bootstrap_ci(result_dir: Path, label: str, T: int, N: int, n_boot: int = 2000, seed: int = 0):
    traj_path = result_dir / "traj.csv"
    with open(result_dir / "results.json") as f:
        res = json.load(f)
    p_true = np.array(res["p_true"])

    with open(traj_path) as f:
        traj_list = [list(map(int, row)) for row in csv.reader(f)]
    traj_arr = np.array(traj_list)
    n = len(traj_arr)
    rng = np.random.default_rng(seed)

    point_estimate = res["rmse"]
    boot_rmses = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_traj = traj_arr[idx]
        p_prime = reconstruct_proportions(sample_traj.tolist(), T, N)
        boot_rmses[b] = rmse(p_true, p_prime)

    ci_low, ci_high = np.percentile(boot_rmses, [2.5, 97.5])
    print(f"  [{label}] point RMSE = {point_estimate:.6f}   "
          f"bootstrap 95% CI = [{ci_low:.6f}, {ci_high:.6f}]   "
          f"(n_boot={n_boot}, std={boot_rmses.std():.6f})")
    return {
        "label": label, "point_rmse": point_estimate,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "boot_std": float(boot_rmses.std()), "n_boot": n_boot,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C = np.array(json.load(f), dtype=float)
    distances, _ = get_distances()
    T, N = C.shape
    solver = SASolver()

    print("=" * 70)
    print("(a) Leave-One-Out Cross Validation (T=12, SA proxy sampler)")
    print("=" * 70)
    loo_summary = loo_cv(C, distances, solver)

    print()
    print("=" * 70)
    print("(b) Sensitivity of F to +-10% perturbation of (alpha, beta)")
    print("=" * 70)
    sens_summary = sensitivity_analysis(C, distances, solver)

    print()
    print("=" * 70)
    print("(c) Bootstrap 95% CI for RMSE (SQA / QA, resampled from existing trajectories)")
    print("=" * 70)
    boot_sqa = bootstrap_ci(Path("results/sqa_30k"), "SQA", T, N)
    boot_qa = bootstrap_ci(Path("results/qa_advantage2_30k"), "QA (repaired)", T, N)

    summary = {
        "loo_cv": loo_summary,
        "sensitivity": sens_summary,
        "bootstrap_ci": [boot_sqa, boot_qa],
    }
    out_path = OUT_DIR / "m5_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()

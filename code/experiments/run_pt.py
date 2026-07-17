"""
Parallel Tempering (replica-exchange Monte Carlo) experiment script.
M4 review response: comparison against a classical sampling baseline.

Round-2 review FB1 response: in response to Prof. Ohzeki's question of
whether the replica selection / temperature-ladder design was fair, the
pipeline was changed to the following (the previous version only reported
samples from the coldest replica, index 0, which turned out to be close to
the worst choice among the 16 replicas).

  1. Collect samples from all 16 replicas (`run_all_replicas`; because the
     exchange method periodically swaps all replicas, intermediate-
     temperature replicas are available at no extra sampling cost).
  2. Select which replica (temperature) to adopt via 10-fold
     cross-validation (to avoid a hindsight selection based on RMSE, the
     best replica is chosen using 9 training folds and evaluated only on
     the held-out fold not used for selection; the same idea as the M5
     (alpha, beta) selection).
  3. Save the full num_reads samples of the replica selected across all
     folds (the consensus) as results.json / sampleset.json / traj.csv, in
     the same format used by the other solvers. The fold-by-fold detail of
     the 10-fold CV (circularity-free held-out metrics) is saved to
     cv_summary.json.

Usage:
    uv run python experiments/run_pt.py
    uv run python experiments/run_pt.py --reads 200 --cv-folds 5  # for a quick smoke test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dimod

from src.flow import build_flow_matrix, reconstruct_proportions
from src.graph import get_distances
from src.metrics import rmse, squared_loss
from src.qubo import build_qubo
from src.solvers.parallel_tempering import run_all_replicas
from src.trajectory import compute_violation_rate, decode_traj, save_results
from src.transition import build_transition_P, _safe_row_normalize

# ===== Parameters (same QUBO settings as SQA/SA) =====
ALPHA = 0.3
BETA = 0.55
LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
SEED = 3
NUM_READS = 30_000
OUTPUT_DIR = Path("results/pt_30k")

# Parallel Tempering-specific settings (same log-spaced ladder as the previous version)
PT_CONFIG = {
    "n_replicas": 16,
    "beta_min": 0.05,
    "beta_max": 60.0,
    "n_sweeps_burn_in": 3000,
    "sweeps_per_sample": 30,
    "swap_interval": 5,
}

K_FOLDS = 10
CV_SEED = 12345  # seed used only for the fold split (independent of the PT run's seed=SEED)


def _rmse_for_indices(traj_all: list, idx: np.ndarray, T: int, N: int, p_true: np.ndarray) -> float:
    subset = [traj_all[i] for i in idx]
    p_prime = reconstruct_proportions(subset, T, N)
    return rmse(p_true, p_prime)


def _select_replica_by_cv(
    traj_by_replica: list[list[list[int]]],
    betas: np.ndarray,
    samples: np.ndarray,
    labels: list,
    T: int,
    N: int,
    Np: int,
    p_true: np.ndarray,
    num_reads: int,
    k_folds: int,
) -> dict:
    """Select a replica (temperature) via 10-fold CV and return the per-fold and aggregated held-out metrics."""
    fold_rng = np.random.default_rng(CV_SEED)
    perm = fold_rng.permutation(num_reads)
    folds = np.array_split(perm, k_folds)

    n_replicas = len(betas)
    test_rmses, test_violations, test_unique_fracs, selected_replicas = [], [], [], []

    for fold_i in range(k_folds):
        heldout_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])

        train_rmse_by_replica = [
            _rmse_for_indices(traj_by_replica[r], train_idx, T, N, p_true)
            for r in range(n_replicas)
        ]
        r_star = int(np.argmin(train_rmse_by_replica))
        selected_replicas.append(r_star)

        test_rmse = _rmse_for_indices(traj_by_replica[r_star], heldout_idx, T, N, p_true)

        heldout_samples = samples[r_star][heldout_idx]
        sampleset = dimod.SampleSet.from_samples(
            (heldout_samples, labels), vartype="BINARY",
            energy=np.zeros(len(heldout_idx)), num_occurrences=np.ones(len(heldout_idx), dtype=int),
        )
        test_violation = compute_violation_rate(sampleset, T, Np)
        heldout_traj = [traj_by_replica[r_star][i] for i in heldout_idx]
        unique_frac = len({tuple(t) for t in heldout_traj}) / len(heldout_traj)

        test_rmses.append(test_rmse)
        test_violations.append(test_violation)
        test_unique_fracs.append(unique_frac)

        print(f"  fold {fold_i:2d}: selected replica={r_star:2d} (beta={betas[r_star]:.4f})  "
              f"held-out: rmse={test_rmse:.4f}  violation_rate={test_violation:.4f}  "
              f"unique_frac={unique_frac:.4f}")

    test_rmses_arr = np.array(test_rmses)
    test_violations_arr = np.array(test_violations)
    test_unique_fracs_arr = np.array(test_unique_fracs)

    # All folds should consistently select the same replica (as reported in
    # the manuscript). As a safeguard, take the mode as the consensus replica.
    consensus_replica = int(np.bincount(selected_replicas).argmax())

    return {
        "k_folds": k_folds,
        "selected_replicas": selected_replicas,
        "selected_betas": [round(float(betas[r]), 6) for r in selected_replicas],
        "consensus_replica": consensus_replica,
        "consensus_beta": round(float(betas[consensus_replica]), 6),
        "unanimous": len(set(selected_replicas)) == 1,
        "test_rmse_per_fold": test_rmses_arr.tolist(),
        "test_violation_per_fold": test_violations_arr.tolist(),
        "test_unique_frac_per_fold": test_unique_fracs_arr.tolist(),
        "test_rmse_mean": float(test_rmses_arr.mean()),
        "test_rmse_std": float(test_rmses_arr.std(ddof=1)) if k_folds > 1 else 0.0,
        "test_violation_mean": float(test_violations_arr.mean()),
        "test_unique_frac_mean": float(test_unique_fracs_arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--cv-folds", type=int, default=K_FOLDS)
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)
    T, N = C_history.shape
    Np = N + 1

    p_true = _safe_row_normalize(C_history)
    distances, _ = get_distances()
    P = build_transition_P(C_history, distances, ALPHA, BETA)
    Q = build_qubo(P, T, LAMBDA_ONEHOT, LAMBDA_P, LAMBDA_DIV, LAMBDA_ENTRY, LAMBDA_MOVE)

    print(f"[PT] Running Parallel Tempering, collecting all {PT_CONFIG['n_replicas']} replicas "
          f"({args.reads} reads each) ...")
    run = run_all_replicas(Q, num_reads=args.reads, seed=SEED, **PT_CONFIG)
    labels = run["labels"]
    betas = run["betas"]
    samples = run["samples"]      # (n_replicas, num_reads, n_vars)
    energies = run["energies"]    # (n_replicas, num_reads)
    n_replicas = samples.shape[0]

    print("[PT] Decoding trajectories for all replicas ...")
    decode_rng = np.random.default_rng(SEED)
    traj_by_replica = [
        [decode_traj(dict(zip(labels, samples[r, s])), T, Np, decode_rng) for s in range(args.reads)]
        for r in range(n_replicas)
    ]

    print(f"\n[PT] Selecting replica temperature via {args.cv_folds}-fold CV "
          f"(circularity-free out-of-sample selection) ...")
    cv = _select_replica_by_cv(
        traj_by_replica, betas, samples, labels, T, N, Np, p_true, args.reads, args.cv_folds,
    )
    r_star = cv["consensus_replica"]
    print(f"\n[PT] Consensus replica={r_star} (beta={betas[r_star]:.4f}), "
          f"unanimous across folds={cv['unanimous']}")
    print(f"[PT] Held-out (circularity-free): RMSE={cv['test_rmse_mean']:.6f} ± {cv['test_rmse_std']:.6f}  "
          f"violation_rate={cv['test_violation_mean']:.6f}  unique_frac={cv['test_unique_frac_mean']:.6f}")

    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    print(f"[PT] CV summary saved to {output_dir}/cv_summary.json")

    # --- Save all samples of the consensus replica, in the same format as the other solvers ---
    num_occurrences = np.ones(args.reads, dtype=int)
    sampleset = dimod.SampleSet.from_samples(
        (samples[r_star], labels), vartype="BINARY",
        energy=energies[r_star], num_occurrences=num_occurrences,
        info={
            "method": "parallel_tempering",
            "n_replicas": n_replicas,
            "betas": betas.tolist(),
            "selected_replica": r_star,
            "selected_beta": float(betas[r_star]),
            "selection_method": f"{args.cv_folds}-fold cross-validation",
            **PT_CONFIG,
        },
    )
    traj_list = traj_by_replica[r_star]
    violation = compute_violation_rate(sampleset, T, Np)
    p_prime = reconstruct_proportions(traj_list, T, N)
    F = build_flow_matrix(traj_list, T, Np)
    loss = squared_loss(p_true, p_prime)
    r = rmse(p_true, p_prime)

    print(f"\n[PT] Selected-replica full-sample metrics (in-sample, all {args.reads} reads): "
          f"violation_rate={violation:.4f}  RMSE={r:.6f}  loss={loss:.6f}")

    save_results(sampleset, traj_list, Q, T, Np, output_dir)

    results = {
        "solver": "PT",
        "alpha": ALPHA,
        "beta": BETA,
        "num_reads": args.reads,
        "num_sweeps": None,
        "seed": SEED,
        "lambda_onehot": LAMBDA_ONEHOT,
        "lambda_P": LAMBDA_P,
        "lambda_div": LAMBDA_DIV,
        "lambda_entry": LAMBDA_ENTRY,
        "lambda_move": LAMBDA_MOVE,
        "violation_rate": round(violation, 6),
        "rmse": round(r, 6),
        "loss": round(loss, 6),
        "C": C_history.tolist(),
        "p_true": p_true.tolist(),
        "p_prime": p_prime.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "pt_config": PT_CONFIG,
        "selected_replica_index": r_star,
        "selected_replica_beta": round(float(betas[r_star]), 6),
        "selection_method": f"{args.cv_folds}-fold cross-validation (see cv_summary.json)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[PT] Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

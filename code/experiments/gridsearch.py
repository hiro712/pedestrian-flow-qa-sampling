"""
alpha/beta grid search + final run with the best parameters.

Usage:
    uv run python experiments/gridsearch.py --solver sqa
    uv run python experiments/gridsearch.py --solver sa --reads 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import get_distances
from src.solvers import get_solver
from experiments._pipeline import run_experiment, run_gridsearch

LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
NUM_SWEEPS = 100   # same setting as in the manuscript
SEED = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["sa", "sqa"], default="sa")
    parser.add_argument("--reads", type=int, default=100, help="number of samples during grid search")
    parser.add_argument("--final-reads", type=int, default=30_000, help="number of samples for the final run")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else Path(f"results/{args.solver}_gridsearch")

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = get_solver(args.solver)

    # Stage 1: wide and coarse
    alphas_s1 = list(map(float, np.arange(0.0, 6.0 + 1e-9, 0.5)))
    betas_s1 = list(map(float, np.arange(0.3, 2.5 + 1e-9, 0.2)))

    print("=== Stage 1 ===")
    a1, b1, _, _ = run_gridsearch(
        C_history=C_history, distances=distances,
        alphas=alphas_s1, betas=betas_s1,
        lambda_onehot=LAMBDA_ONEHOT, lambda_P=LAMBDA_P, lambda_div=LAMBDA_DIV,
        lambda_entry=LAMBDA_ENTRY, lambda_move=LAMBDA_MOVE,
        solver=solver, num_reads=args.reads, num_sweeps=NUM_SWEEPS, seed=SEED,
        output_dir=out_root, solver_name=args.solver.upper(),
        history_csv="history_stage1.csv",
    )

    # Stage 2: fine-grained around the best point
    win_a, step_a = 0.6, 0.1
    win_b, step_b = 0.6, 0.05
    alphas_s2 = list(map(float, np.arange(max(0.0, a1 - win_a), a1 + win_a + 1e-9, step_a)))
    betas_s2 = list(map(float, np.arange(max(0.1, b1 - win_b), b1 + win_b + 1e-9, step_b)))

    print(f"\n=== Stage 2 (around α={a1}, β={b1}) ===")
    a2, b2, _, _ = run_gridsearch(
        C_history=C_history, distances=distances,
        alphas=alphas_s2, betas=betas_s2,
        lambda_onehot=LAMBDA_ONEHOT, lambda_P=LAMBDA_P, lambda_div=LAMBDA_DIV,
        lambda_entry=LAMBDA_ENTRY, lambda_move=LAMBDA_MOVE,
        solver=solver, num_reads=args.reads, num_sweeps=NUM_SWEEPS, seed=SEED,
        output_dir=out_root, solver_name=args.solver.upper(),
        history_csv="history_stage2.csv",
    )

    # Final run
    print(f"\n=== Final run: α={a2}, β={b2}, reads={args.final_reads} ===")
    run_experiment(
        C_history=C_history, distances=distances,
        alpha=a2, beta=b2,
        lambda_onehot=LAMBDA_ONEHOT, lambda_P=LAMBDA_P, lambda_div=LAMBDA_DIV,
        lambda_entry=LAMBDA_ENTRY, lambda_move=LAMBDA_MOVE,
        solver=get_solver(args.solver),
        num_reads=args.final_reads, num_sweeps=NUM_SWEEPS, seed=SEED,
        output_dir=out_root / "final",
        solver_name=f"{args.solver.upper()}_final",
    )


if __name__ == "__main__":
    main()

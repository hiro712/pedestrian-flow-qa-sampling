"""
Ablation of the transition-preference term P (SQA).

Keeps the QUBO, the constraints and the SQA sampler exactly as in the main experiment
and only disables the preference term, to show how far the reconstructed flow matrix F
(i.e. the ranked corridors used for sign placement) moves while the RMSE barely changes.

Conditions:
    A  (alpha, beta) = (0.30, 0.55)   settings reported in the manuscript (control)
    B  (alpha, beta) = (0.0, 0.0)     P uniform over all nodes: neither distance nor popularity
    C  lambda_P = 0                   preference term removed from the QUBO

Conditions A and C are the ones reported in the manuscript, in the ablation table of
"Robustness and validation checks" (RMSE 0.0773 -> 0.0810). Condition B is kept
here for completeness but is not a valid comparison: the lambda weights were tuned jointly,
so replacing P alone unbalances them, the uniqueness constraint breaks down (~82% violation
rate) and the randomized repair drives p' towards the uniform distribution, which lowers the
RMSE for reasons unrelated to model quality.

SEED is None on purpose: passing a seed to OpenJij's SQASampler makes all reads return the
same trajectory, which removes the sample diversity the method relies on. Reported values
therefore vary between runs by about +/-0.0004.

Usage:
    uv run python experiments/run_sqa_ablation.py
    uv run python experiments/run_sqa_ablation.py --reads 100  # for a quick smoke test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import get_distances
from src.solvers.sqa import SQASolver
from experiments._pipeline import run_experiment

# ===== Parameters (identical to experiments/run_sqa.py unless overridden per condition) =====
ALPHA = 0.3
BETA = 0.55
LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
SEED = None
NUM_READS = 30_000   # same setting as in the manuscript
NUM_SWEEPS = 100     # same setting as in the manuscript (note: not OpenJij's default of 1000)
OUTPUT_DIR = Path("results/sqa_ablation")

BASE = dict(
    lambda_onehot=LAMBDA_ONEHOT,
    lambda_P=LAMBDA_P,
    lambda_div=LAMBDA_DIV,
    lambda_entry=LAMBDA_ENTRY,
    lambda_move=LAMBDA_MOVE,
)

CONDITIONS = [
    ("A", "as in the manuscript", dict(alpha=ALPHA, beta=BETA, **BASE)),
    ("B", "P uniform (alpha=beta=0)", dict(alpha=0.0, beta=0.0, **BASE)),
    ("C", "preference term removed (lambda_P=0)",
     dict(alpha=ALPHA, beta=BETA, **{**BASE, "lambda_P": 0.0})),
]

# Edges of the venue graph (Fig. 1b); corridor ratios are reported on these links only.
EDGES = [tuple(sorted(e)) for e in
         [(8, 9), (8, 0), (0, 10), (8, 6), (6, 7), (7, 5), (5, 4), (4, 1), (1, 3), (3, 2)]]


def top_corridors(F: np.ndarray, k: int = 3) -> list[tuple[tuple[int, int], float]]:
    """Flow ratio (%) per venue-graph edge, highest first."""
    total = F.sum()
    ratio = {e: (F[e[0], e[1]] + F[e[1], e[0]]) / total * 100 for e in EDGES}
    return sorted(ratio.items(), key=lambda x: -x[1])[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS, help="number of samples (default: 30000)")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--conditions", type=str, default="AC",
                        help="which conditions to run, e.g. 'AC' (default) or 'ABC'")
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()

    print(f"reads={args.reads}\n")
    for key, label, cfg in CONDITIONS:
        if key not in args.conditions:
            continue
        result = run_experiment(
            C_history=C_history,
            distances=distances,
            solver=SQASolver(),
            num_reads=args.reads,
            num_sweeps=NUM_SWEEPS,
            seed=SEED,
            output_dir=Path(args.out) / key,
            solver_name="SQA",
            **cfg,
        )
        F = np.array(result["F"], dtype=float)
        top = ", ".join(f"{a}-{b}:{v:.2f}%" for (a, b), v in top_corridors(F))
        print(f"{key} {label:38s} RMSE={result['rmse']:.4f}  "
              f"violation={result.get('violation_rate', 0):.4f}  top3={top}")


if __name__ == "__main__":
    main()

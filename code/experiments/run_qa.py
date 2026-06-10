"""
D-Wave 実機 QA 実験スクリプト。

使い方:
    uv run python experiments/run_qa.py
    uv run python experiments/run_qa.py --reads 100 --annealing-time 20

事前準備:
    .env.local に DWAVE_SOLVER_NAME と DWAVE_API_TOKEN を設定すること。
    利用可能なソルバー: uv run python -c "from dwave.cloud import Client; print([s.id for s in Client.from_config().get_solvers()])"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv(Path(__file__).resolve().parents[1] / ".env.local")

from src.graph import get_distances
from src.solvers.qa import QASolver
from experiments._pipeline import run_experiment

ALPHA = 0.3
BETA = 0.55
LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
SEED = None
NUM_READS = 30_000
ANNEALING_TIME = 20  # μs
OUTPUT_DIR = Path("results/qa_30k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--annealing-time", type=int, default=ANNEALING_TIME, dest="at")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    solver_name = os.getenv("DWAVE_SOLVER_NAME", "unknown")
    print(f"D-Wave solver: {solver_name}")

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = QASolver()

    # annealing_time は QASolver の solve() で sample_config に渡す
    # _pipeline.run_experiment は num_reads と seed しか cfg に追加しないため、
    # ここでは SolverBase を wrap して annealing_time を注入する
    class _QAWithAT(type(solver)):
        def solve(self, Q, sample_config=None):
            cfg = dict(sample_config or {})
            cfg["annealing_time"] = args.at
            return super().solve(Q, sample_config=cfg)

    solver.__class__ = _QAWithAT

    out_dir = Path(args.out)
    run_experiment(
        C_history=C_history,
        distances=distances,
        alpha=ALPHA,
        beta=BETA,
        lambda_onehot=LAMBDA_ONEHOT,
        lambda_P=LAMBDA_P,
        lambda_div=LAMBDA_DIV,
        lambda_entry=LAMBDA_ENTRY,
        lambda_move=LAMBDA_MOVE,
        solver=solver,
        num_reads=args.reads,
        seed=SEED,
        output_dir=out_dir,
        solver_name=f"QA({solver_name})",
    )


if __name__ == "__main__":
    main()

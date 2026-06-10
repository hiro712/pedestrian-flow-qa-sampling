"""
Parallel Tempering（交換モンテカルロ法）実験スクリプト。
M4 査読対応: 古典的サンプリングベースラインとの比較。

使い方:
    uv run python experiments/run_pt.py
    uv run python experiments/run_pt.py --reads 200  # 動作確認用
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import get_distances
from src.solvers.parallel_tempering import ParallelTemperingSolver
from experiments._pipeline import run_experiment

# ===== パラメータ（SQA/SA と同一の QUBO 設定） =====
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

# Parallel Tempering 固有設定
PT_CONFIG = {
    "n_replicas": 16,
    "beta_min": 0.05,
    "beta_max": 60.0,
    "n_sweeps_burn_in": 3000,
    "sweeps_per_sample": 30,
    "swap_interval": 5,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = ParallelTemperingSolver()

    # run_experiment は sample_config に num_reads/num_sweeps/seed を merge するため、
    # PT固有設定をソルバーの default_config に注入しておく
    solver.default_config = {**solver.default_config, **PT_CONFIG, "seed": SEED}

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
        num_sweeps=None,
        seed=SEED,
        output_dir=Path(args.out),
        solver_name="PT",
    )


if __name__ == "__main__":
    main()

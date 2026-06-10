"""
SQA (Simulated Quantum Annealing) 実験スクリプト。

使い方:
    uv run python experiments/run_sqa.py
    uv run python experiments/run_sqa.py --reads 100  # 動作確認用
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import get_distances
from src.solvers.sqa import SQASolver
from experiments._pipeline import run_experiment

# ===== パラメータ =====
ALPHA = 0.3
BETA = 0.55
LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
SEED = None
NUM_READS = 30_000   # 論文と同じ設定
NUM_SWEEPS = 100     # 論文と同じ設定（OpenJij デフォルト 1000 ではないので注意）
OUTPUT_DIR = Path("results/sqa_30k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS, help="サンプル数 (default: 30000)")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR), help="出力ディレクトリ")
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = SQASolver()

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
        num_sweeps=NUM_SWEEPS,
        seed=SEED,
        output_dir=Path(args.out),
        solver_name="SQA",
    )


if __name__ == "__main__":
    main()

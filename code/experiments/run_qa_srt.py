"""
M2(a): Spin-Reversal Transform (SRT) 実験。

ゲージ依存ノイズ（積分制御誤差の一部）を、ランダムなゲージ変換を多数適用して
平均化することで除去した場合に、制約違反率がどう変化するかを調べる。

使い方:
    uv run python experiments/run_qa_srt.py
    uv run python experiments/run_qa_srt.py --reads 30000 --srt 100
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

from dwave.preprocessing import SpinReversalTransformComposite
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from src.graph import get_distances
from src.solvers.base import SolverBase
from experiments._pipeline import run_experiment
from experiments._embedding_cache import get_or_compute_embedding

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
NUM_SRT = 100
OUTPUT_DIR = Path("results/qa_srt_30k")
EMBEDDING_CACHE = Path("results/_embedding_cache/qubo_main_embedding.json")


class SRTQASolver(SolverBase):
    """FixedEmbeddingComposite + SpinReversalTransformComposite でゲージ平均化したQA。

    SRTは係数の符号を反転するだけで変数間の結合構造(=埋め込み)は変えないため、
    埋め込みを1回だけ計算してFixedEmbeddingCompositeで固定し、100ゲージすべてで
    使い回す(EmbeddingCompositeだとゲージごとに埋め込み探索が走り直してしまう)。
    """

    def __init__(self, num_spin_reversal_transforms: int, annealing_time: int,
                 embedding_cache: Path = EMBEDDING_CACHE):
        solver_name = os.getenv("DWAVE_SOLVER_NAME")
        token = os.getenv("DWAVE_API_TOKEN")
        self.hw_sampler = DWaveSampler(solver=solver_name, token=token)
        self.embedding_cache = embedding_cache
        self.sampler = None
        self.nsrt = num_spin_reversal_transforms
        self.at = annealing_time

    def solve(self, Q, sample_config=None):
        cfg = self._merged_config(sample_config)
        num_reads = cfg.pop("num_reads", 10)
        cfg.pop("seed", None)

        if self.sampler is None:
            embedding = get_or_compute_embedding(dict(Q), self.hw_sampler, self.embedding_cache)
            fixed = FixedEmbeddingComposite(self.hw_sampler, embedding)
            self.sampler = SpinReversalTransformComposite(fixed)

        return self.sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            num_spin_reversal_transforms=self.nsrt,
            annealing_time=self.at,
            **cfg,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--srt", type=int, default=NUM_SRT, help="num_spin_reversal_transforms")
    parser.add_argument("--annealing-time", type=int, default=ANNEALING_TIME, dest="at")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    solver_name = os.getenv("DWAVE_SOLVER_NAME", "unknown")
    print(f"D-Wave solver: {solver_name}  (SRT gauges = {args.srt}, annealing_time = {args.at} us)")

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = SRTQASolver(num_spin_reversal_transforms=args.srt, annealing_time=args.at)

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
        solver_name=f"QA-SRT{args.srt}({solver_name})",
    )


if __name__ == "__main__":
    main()

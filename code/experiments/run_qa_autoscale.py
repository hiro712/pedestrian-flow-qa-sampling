"""
M1(b): auto_scale=False + 手動スケーリング実験。

QUBO の係数を、埋め込み後の Ising 表現の h_range / J_range いっぱいまで
線形スケーリングしてから auto_scale=False で投入する。これにより最小係数を
ICE ノイズフロア (~2σ) より上に保てるかどうかを検証する。

使い方:
    uv run python experiments/run_qa_autoscale.py
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

from dimod import BinaryQuadraticModel, concatenate
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from src.graph import get_distances
from src.qubo import build_qubo
from src.solvers.base import SolverBase
from src.transition import build_transition_P, _safe_row_normalize
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
OUTPUT_DIR = Path("results/qa_autoscale_30k")
EMBEDDING_CACHE = Path("results/_embedding_cache/qubo_main_embedding.json")
MAX_READS_PER_CALL = 5000  # ソルバー側のnum_reads上限(現行 [1, 10000])に収まるチャンクサイズ。src/solvers/qa.pyに合わせる


class ManualScaleQASolver(SolverBase):
    """係数を h_range/J_range いっぱいまでスケールし、auto_scale=False で投入するQA。

    SRT実験(run_qa_srt.py)と同一QUBO構造のため、同じ埋め込みキャッシュを共有して
    使い回す(埋め込み探索の重複計算を避け、両実験の比較条件を揃える)。
    """

    def __init__(self, annealing_time: int, embedding_cache: Path = EMBEDDING_CACHE):
        solver_name = os.getenv("DWAVE_SOLVER_NAME")
        token = os.getenv("DWAVE_API_TOKEN")
        self.hw_sampler = DWaveSampler(solver=solver_name, token=token)
        self.embedding_cache = embedding_cache
        self.sampler = None
        self.at = annealing_time
        self.scale_info: dict = {}

    def _scale_factor(self, Q: dict) -> float:
        bqm = BinaryQuadraticModel.from_qubo(Q)
        h, J, _ = bqm.to_ising()
        vals = np.array([abs(v) for v in list(h.values()) + list(J.values()) if v != 0.0])
        max_val = float(vals.max())
        min_val = float(vals.min())

        h_range = self.hw_sampler.properties.get("h_range", [-2.0, 2.0])
        j_range = self.hw_sampler.properties.get("j_range", [-1.0, 1.0])
        allowed_max = min(abs(h_range[0]), abs(h_range[1]), abs(j_range[0]), abs(j_range[1]))

        safety = 0.95
        scale = safety * allowed_max / max_val

        self.scale_info = {
            "max_abs_coeff_before": max_val,
            "min_abs_coeff_before": min_val,
            "ratio_before": max_val / min_val,
            "h_range": h_range,
            "j_range": j_range,
            "allowed_max": allowed_max,
            "scale_factor": scale,
            "max_abs_coeff_after": max_val * scale,
            "min_abs_coeff_after": min_val * scale,
        }
        return scale

    def solve(self, Q, sample_config=None):
        cfg = self._merged_config(sample_config)
        num_reads = cfg.pop("num_reads", 10)
        cfg.pop("seed", None)

        if self.sampler is None:
            embedding = get_or_compute_embedding(dict(Q), self.hw_sampler, self.embedding_cache)
            self.sampler = FixedEmbeddingComposite(self.hw_sampler, embedding)

        scale = self._scale_factor(dict(Q))
        Q_scaled = {k: v * scale for k, v in Q.items()}

        print(f"[ManualScale] scale_factor={scale:.6f}  "
              f"min|coeff| {self.scale_info['min_abs_coeff_before']:.6f} -> {self.scale_info['min_abs_coeff_after']:.6f}  "
              f"max|coeff| {self.scale_info['max_abs_coeff_before']:.6f} -> {self.scale_info['max_abs_coeff_after']:.6f}")

        if num_reads <= MAX_READS_PER_CALL:
            return self.sampler.sample_qubo(
                Q_scaled,
                num_reads=num_reads,
                auto_scale=False,
                annealing_time=self.at,
                **cfg,
            )

        # ソルバー側のnum_reads上限を超える場合はチャンクに分割して送信し、結果をconcatenateする
        chunks = []
        remaining = num_reads
        while remaining > 0:
            n = min(remaining, MAX_READS_PER_CALL)
            chunks.append(n)
            remaining -= n

        sets = []
        for n in chunks:
            sets.append(self.sampler.sample_qubo(
                Q_scaled,
                num_reads=n,
                auto_scale=False,
                annealing_time=self.at,
                **cfg,
            ))
        return concatenate(sets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--annealing-time", type=int, default=ANNEALING_TIME, dest="at")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    solver_name = os.getenv("DWAVE_SOLVER_NAME", "unknown")
    print(f"D-Wave solver: {solver_name}  (auto_scale=False, annealing_time = {args.at} us)")

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)

    distances, _ = get_distances()
    solver = ManualScaleQASolver(annealing_time=args.at)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
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
        solver_name=f"QA-ManualScale({solver_name})",
    )

    # スケーリング情報も保存
    with open(out_dir / "scale_info.json", "w", encoding="utf-8") as f:
        json.dump(solver.scale_info, f, ensure_ascii=False, indent=2)
    print(f"Scale info saved to {out_dir / 'scale_info.json'}")


if __name__ == "__main__":
    main()

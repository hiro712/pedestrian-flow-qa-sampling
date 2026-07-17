"""
Embedding cache shared by the M1/M2 experiments.

The SRT experiment (run_qa_srt.py) and the auto_scale experiment
(run_qa_autoscale.py) use the identical QUBO structure (same alpha/beta/lambda
settings), so the minor-embedding is computed once, saved to JSON, and reused
by both experiments. A spin-reversal transform (SRT) only flips coefficient
signs and does not change the coupling structure between variables, so the
same embedding can be reused as-is.

Usage:
    embedding = get_or_compute_embedding(Q, hw_sampler, cache_path)
    fixed_sampler = FixedEmbeddingComposite(hw_sampler, embedding)
"""

from __future__ import annotations

import json
from pathlib import Path

from dimod import BinaryQuadraticModel
from minorminer import find_embedding


def get_or_compute_embedding(Q: dict, hw_sampler, cache_path: Path) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        embedding = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in raw.items()}
        print(f"[embedding] cached embedding loaded from {cache_path} "
              f"({len(embedding)} logical variables)")
        return embedding

    bqm = BinaryQuadraticModel.from_qubo(Q)
    source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
    target_edgelist = hw_sampler.edgelist

    print(f"[embedding] no cache found at {cache_path}; computing minor-embedding "
          f"for {bqm.num_variables} logical variables (this is a one-time, CPU-only step "
          f"and may take many minutes for a dense QUBO)…")
    embedding = find_embedding(source_edgelist, target_edgelist)
    if not embedding:
        raise RuntimeError("no embedding found")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(embedding, f)
    print(f"[embedding] computed embedding for {len(embedding)} logical variables, "
          f"saved to {cache_path}")
    return embedding

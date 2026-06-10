"""
M1/M2 実験で共有する埋め込み(embedding)のキャッシュ。

SRT実験(run_qa_srt.py)とauto_scale実験(run_qa_autoscale.py)は同一のQUBO構造
(同一のalpha/beta/lambda設定)を使うため、マイナー埋め込みを1回だけ計算して
JSONに保存し、両実験で使い回す。スピン反転変換(SRT)も係数の符号を変えるだけで
変数間の結合構造は変えないため、同じ埋め込みをそのまま使い回せる。

使い方:
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

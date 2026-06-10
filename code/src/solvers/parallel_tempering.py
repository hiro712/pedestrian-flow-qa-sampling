"""
M4 査読対応: 古典的サンプリングベースラインとしての Parallel Tempering
（交換モンテカルロ法 / レプリカ交換法）。

複数の逆温度 beta を持つレプリカを並行してシングルスピンフリップ
Metropolis 更新で走らせ、隣接レプリカ間で周期的に状態を交換することで
低温分布からの効率的なサンプリングを実現する古典的手法。

QUBO E(x) = x^T W x （W は対称行列、対角=線形項）に対し、
1スピンフリップのエネルギー差分は

    dE_k = W[k,k] + 2*(1 - 2*x_k) * (W @ x)[k]

で閉形式に求まるため、場ベクトル f = W @ x を逐次更新しながら
全レプリカを ndarray でベクトル化して同時に更新する。
"""

from __future__ import annotations

from typing import cast

import dimod
import numpy as np
from dimod import SampleSet

from src.solvers.base import Qubo, SampleConfig, SolverBase


def _qubo_to_dense(Q: Qubo) -> tuple[np.ndarray, list]:
    """QUBO 辞書を対称行列 W (E(x) = x^T W x) と変数ラベル一覧に変換する。"""
    labels = sorted({u for k in Q for u in k} | {v for k in Q for v in k})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    W = np.zeros((n, n))
    for (u, v), w in Q.items():
        iu, iv = label_to_idx[u], label_to_idx[v]
        if iu == iv:
            W[iu, iu] += w
        else:
            W[iu, iv] += w / 2.0
            W[iv, iu] += w / 2.0
    return W, labels


class ParallelTemperingSolver(SolverBase):
    """
    Parameters (sample_config 経由で指定可能)
    ----------
    num_reads : int
        収集する総サンプル数
    n_replicas : int
        レプリカ数（既定 8）
    beta_min, beta_max : float
        逆温度ラダーの範囲（対数等間隔; 既定 0.01 — 50.0）
    n_sweeps_burn_in : int
        サンプル収集前のバーンイン掃引数
    sweeps_per_sample : int
        サンプル間の間引き掃引数（自己相関低減）
    swap_interval : int
        レプリカ交換を試みる間隔（掃引数）
    seed : int | None
    """

    default_config: dict = {
        "num_reads": 1000,
        "n_replicas": 8,
        "beta_min": 0.01,
        "beta_max": 50.0,
        "n_sweeps_burn_in": 200,
        "sweeps_per_sample": 2,
        "swap_interval": 5,
        "seed": None,
    }

    def solve(self, Q: Qubo, sample_config: SampleConfig | None = None) -> SampleSet:
        cfg = self._merged_config(sample_config)
        num_reads = int(cfg["num_reads"])
        n_replicas = int(cfg["n_replicas"])
        beta_min = float(cfg["beta_min"])
        beta_max = float(cfg["beta_max"])
        n_burn = int(cfg["n_sweeps_burn_in"])
        thin = int(cfg["sweeps_per_sample"])
        swap_interval = int(cfg["swap_interval"])
        seed = cfg.get("seed")

        rng = np.random.default_rng(seed)
        W, labels = _qubo_to_dense(Q)
        n = len(labels)
        diagW = np.diag(W).copy()

        # レプリカ初期化（ランダム binary 状態）
        X = rng.integers(0, 2, size=(n_replicas, n)).astype(np.int8)
        F = X.astype(float) @ W  # f_r = W @ x_r  (n_replicas, n)

        # 逆温度ラダー（対数等間隔; index 0 = 最も低温=高 beta）
        betas = np.geomspace(beta_max, beta_min, n_replicas)

        def energies() -> np.ndarray:
            return np.sum(X * F, axis=1)

        def sweep() -> None:
            nonlocal X, F
            order = rng.permutation(n)
            for k in order:
                xk = X[:, k].astype(float)
                dE = diagW[k] + 2.0 * (1.0 - 2.0 * xk) * F[:, k]
                accept_prob = np.exp(np.clip(-betas * dE, -700, 700))
                u = rng.random(n_replicas)
                flip = u < accept_prob
                if not flip.any():
                    continue
                delta = np.where(flip, 1.0 - 2.0 * xk, 0.0)  # +1 (0->1) or -1 (1->0) where flipped
                X[flip, k] = 1 - X[flip, k]
                F += delta[:, None] * W[k, :][None, :]

        def attempt_swaps() -> None:
            nonlocal X, F
            E = energies()
            # 隣接レプリカ間で交換 (i, i+1), i = 0,2,4,... と 1,3,5,... を交互に
            for offset in (0, 1):
                for i in range(offset, n_replicas - 1, 2):
                    j = i + 1
                    d = (betas[i] - betas[j]) * (E[i] - E[j])
                    if rng.random() < np.exp(min(0.0, d)):
                        X[[i, j]] = X[[j, i]]
                        F[[i, j]] = F[[j, i]]
                        E[[i, j]] = E[[j, i]]

        # --- バーンイン ---
        for sweep_idx in range(n_burn):
            sweep()
            if sweep_idx % swap_interval == 0:
                attempt_swaps()

        # --- サンプル収集（最も低温=ターゲット分布のレプリカ index 0 から）---
        collected = []
        collected_E = []
        sweep_idx = 0
        while len(collected) < num_reads:
            for _ in range(thin):
                sweep()
                sweep_idx += 1
                if sweep_idx % swap_interval == 0:
                    attempt_swaps()
            collected.append(X[0].copy())
            collected_E.append(float(energies()[0]))

        samples = np.array(collected[:num_reads], dtype=np.int8)
        sample_energies = np.array(collected_E[:num_reads], dtype=float)
        num_occurrences = np.ones(len(samples), dtype=int)

        sampleset = dimod.SampleSet.from_samples(
            (samples, labels), vartype="BINARY",
            energy=sample_energies, num_occurrences=num_occurrences,
            info={
                "method": "parallel_tempering",
                "n_replicas": n_replicas,
                "betas": betas.tolist(),
                "n_sweeps_burn_in": n_burn,
                "sweeps_per_sample": thin,
                "swap_interval": swap_interval,
            },
        )
        return cast(SampleSet, sampleset)

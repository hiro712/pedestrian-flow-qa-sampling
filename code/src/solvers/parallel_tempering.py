"""
M4 review response: Parallel Tempering (replica-exchange Monte Carlo) as a
classical sampling baseline.

Multiple replicas, each held at a different inverse temperature beta, are run
in parallel with single-spin-flip Metropolis updates; adjacent replicas
periodically exchange states, which is the classical technique for efficient
sampling from a low-temperature distribution.

For a QUBO E(x) = x^T W x (W symmetric, diagonal = linear terms), the
single-spin-flip energy difference has the closed form

    dE_k = W[k,k] + 2*(1 - 2*x_k) * (W @ x)[k]

so we incrementally update the field vector f = W @ x and vectorize all
replicas together as an ndarray.

Round-2 review FB1 response (to Prof. Ohzeki's question of whether the
replica selection / temperature-ladder design was fair): because PT
periodically exchanges all replicas, the intermediate-temperature replicas
are available at no extra cost beyond the coldest one (index 0).
`run_all_replicas()` exists to examine this: it saves samples from every
replica during the collection step (using the same dynamics as
`ParallelTemperingSolver.solve()`). `experiments/run_pt.py` uses it to
evaluate all replicas and select a replica temperature via 10-fold CV (so
the choice isn't made with hindsight).
"""

from __future__ import annotations

from typing import cast

import dimod
import numpy as np
from dimod import SampleSet

from src.solvers.base import Qubo, SampleConfig, SolverBase


def _qubo_to_dense(Q: Qubo) -> tuple[np.ndarray, list]:
    """Convert a QUBO dict into a symmetric matrix W (E(x) = x^T W x) and a list of variable labels."""
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


def run_all_replicas(
    Q: Qubo,
    *,
    num_reads: int = 30_000,
    n_replicas: int = 16,
    beta_min: float = 0.05,
    beta_max: float = 60.0,
    n_sweeps_burn_in: int = 3000,
    sweeps_per_sample: int = 30,
    swap_interval: int = 5,
    seed: int | None = 3,
) -> dict:
    """
    Run Parallel Tempering once and collect samples from every replica.

    Uses the same dynamics (sweep / replica exchange) as
    `ParallelTemperingSolver.solve()`. The only difference is that the
    collection step saves all replicas `X[:]`, not just the coldest
    replica `X[0]`.

    Returns
    -------
    dict with keys:
        "labels": list of variable labels
        "betas": (n_replicas,) inverse temperature of each replica (index 0 is coldest = beta_max)
        "samples": (n_replicas, num_reads, n_vars) array of 0/1 values
        "energies": (n_replicas, num_reads) array of energies
    """
    rng = np.random.default_rng(seed)
    W, labels = _qubo_to_dense(Q)
    n = len(labels)
    diagW = np.diag(W).copy()

    X = rng.integers(0, 2, size=(n_replicas, n)).astype(np.int8)
    F = X.astype(float) @ W

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
            delta = np.where(flip, 1.0 - 2.0 * xk, 0.0)
            X[flip, k] = 1 - X[flip, k]
            F += delta[:, None] * W[k, :][None, :]

    def attempt_swaps() -> None:
        nonlocal X, F
        E = energies()
        for offset in (0, 1):
            for i in range(offset, n_replicas - 1, 2):
                j = i + 1
                d = (betas[i] - betas[j]) * (E[i] - E[j])
                if rng.random() < np.exp(min(0.0, d)):
                    X[[i, j]] = X[[j, i]]
                    F[[i, j]] = F[[j, i]]
                    E[[i, j]] = E[[j, i]]

    # --- Burn-in ---
    for sweep_idx in range(n_sweeps_burn_in):
        sweep()
        if sweep_idx % swap_interval == 0:
            attempt_swaps()

    # --- Sample collection (all replicas) ---
    collected = np.empty((num_reads, n_replicas, n), dtype=np.int8)
    collected_E = np.empty((num_reads, n_replicas), dtype=float)
    sweep_idx = 0
    for r in range(num_reads):
        for _ in range(sweeps_per_sample):
            sweep()
            sweep_idx += 1
            if sweep_idx % swap_interval == 0:
                attempt_swaps()
        collected[r] = X
        collected_E[r] = energies()

    # (num_reads, n_replicas, n) -> (n_replicas, num_reads, n)
    samples = np.transpose(collected, (1, 0, 2))
    sample_energies = np.transpose(collected_E, (1, 0))

    return {
        "labels": labels,
        "betas": betas,
        "samples": samples,
        "energies": sample_energies,
    }


class ParallelTemperingSolver(SolverBase):
    """
    Parameters (settable via sample_config)
    ----------
    num_reads : int
        Total number of samples to collect
    n_replicas : int
        Number of replicas (default 8)
    beta_min, beta_max : float
        Range of the inverse-temperature ladder (log-spaced; default 0.01 - 50.0)
    n_sweeps_burn_in : int
        Number of burn-in sweeps before sample collection
    sweeps_per_sample : int
        Number of thinning sweeps between samples (to reduce autocorrelation)
    swap_interval : int
        Interval (in sweeps) at which replica exchange is attempted
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

        # Initialize replicas (random binary state)
        X = rng.integers(0, 2, size=(n_replicas, n)).astype(np.int8)
        F = X.astype(float) @ W  # f_r = W @ x_r  (n_replicas, n)

        # Inverse-temperature ladder (log-spaced; index 0 = coldest = highest beta)
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
            # Exchange adjacent replicas (i, i+1), alternating between i = 0,2,4,... and 1,3,5,...
            for offset in (0, 1):
                for i in range(offset, n_replicas - 1, 2):
                    j = i + 1
                    d = (betas[i] - betas[j]) * (E[i] - E[j])
                    if rng.random() < np.exp(min(0.0, d)):
                        X[[i, j]] = X[[j, i]]
                        F[[i, j]] = F[[j, i]]
                        E[[i, j]] = E[[j, i]]

        # --- Burn-in ---
        for sweep_idx in range(n_burn):
            sweep()
            if sweep_idx % swap_interval == 0:
                attempt_swaps()

        # --- Sample collection (from the coldest = target-distribution replica, index 0) ---
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

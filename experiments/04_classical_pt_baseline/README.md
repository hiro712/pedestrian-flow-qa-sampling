# 04 — Classical sampling baseline: Parallel Tempering

**Why this experiment exists:** the manuscript argues that QA's stochasticity is a *useful sampling
resource* — not just a noisy optimizer. That claim is only meaningful relative to how a strong
classical stochastic sampler behaves on the same problem, so this experiment runs replica-exchange
Monte Carlo (Parallel Tempering) as a matched classical baseline.

**Scripts:**
- `code/src/solvers/parallel_tempering.py` — replica-exchange Monte Carlo (16 replicas,
  beta in [0.05, 60], 3,000-sweep burn-in)
- `code/experiments/run_pt.py`

**Data:** `data/observations.json` (identical QUBO, weights, and (alpha, beta) as all other solvers)
**Run:**
```bash
uv run python experiments/run_pt.py
```

**Results:** `results/pt_30k/` — RMSE = 0.191462, violation_rate = 0.0,
**only 24 unique trajectories out of 30,000 samples** (compare SQA's 29,999/30,000);
`sampleset.json.gz`/`traj.csv.gz` contain every individual sample and decoded trajectory
(gzip-compressed)

**Interpretation:** PT respects the hard uniqueness constraint essentially perfectly — like SQA —
but its retained ensemble collapses onto a handful of trajectories: it satisfies constraints but
fails to explore the feasible manifold. Together with folders 01 and 03, this gives a three-way
contrast that is central to the paper's argument:
- SQA: feasible **and** diverse
- hardware QA: diverse but largely infeasible (~95% violation)
- PT: feasible but not diverse (mode-collapsed)

This demonstrates that "useful sampling of the feasible manifold" is a non-trivial joint property
that neither classical heuristic achieves for free.

# 04 — Classical sampling baseline: Parallel Tempering

**Why this experiment exists:** the manuscript argues that QA's stochasticity is a *useful sampling
resource* — not just a noisy optimizer. That claim is only meaningful relative to how a strong
classical stochastic sampler behaves on the same problem, so this experiment runs replica-exchange
Monte Carlo (Parallel Tempering) as a matched classical baseline.

**Scripts:**
- `code/src/solvers/parallel_tempering.py` — replica-exchange Monte Carlo dynamics (16 replicas,
  beta in [0.05, 60], 3,000-sweep burn-in). `ParallelTemperingSolver` collects only the coldest
  replica (index 0); `run_all_replicas()` runs the identical dynamics but collects samples from
  **all** 16 replicas, since PT's exchange moves make every replica available at no extra
  sampling cost.
- `code/experiments/run_pt.py` — runs `run_all_replicas()`, then selects which replica
  (temperature) to report via 10-fold cross-validation (see "Methodology" below), and saves the
  selected replica's full sample set in the same `results.json`/`sampleset.json.gz`/`traj.csv.gz`
  format used by the other solvers.

**Data:** `data/observations.json` (identical QUBO, weights, and (alpha, beta) as all other solvers)
**Run:**
```bash
uv run python experiments/run_pt.py
```

**Methodology — replica/temperature selection:** an earlier version of this experiment reported
only the coldest replica (index 0, beta=60), which turned out to be close to the worst of the 16
available replicas for this task (violation_rate=0.0, RMSE=0.191, only 24/30,000 unique
trajectories — a mode-collapsed ensemble). Because PT couples all 16 replicas via periodic
exchange moves, every replica's samples are available for free; evaluating each one individually
shows reconstruction quality improving with temperature up to an intermediate replica, then
degrading again. To select a replica without circularity (i.e. without picking the one that
happens to minimize RMSE against the very data used to evaluate it), we run 10-fold
cross-validation over the 30,000 retained samples: for each fold, the replica minimizing RMSE on
the other 9 folds is selected, then evaluated only on the held-out fold. All 10 folds
unanimously select the same replica (index 9, beta≈0.852), and its held-out performance matches
its in-sample performance closely (no overfitting to the fold split).

**Results:** `results/pt_30k/`
- `results.json` — full run at the selected replica (beta=0.852398): violation_rate=0.0057,
  RMSE=0.06899, using all 30,000 samples from that replica (in-sample).
- `cv_summary.json` — the 10-fold cross-validation detail: fold-by-fold selected replica/beta and
  held-out metrics, plus the aggregated **circularity-free** estimate: RMSE=0.069184±0.000511,
  violation_rate≈0.57%, ≈99.99% of held-out trajectories unique. This is the number reported in
  the manuscript.
- `sampleset.json.gz`/`traj.csv.gz` — every individual sample and decoded trajectory from the
  selected replica (gzip-compressed).

**Interpretation:** with its temperature suitably chosen, PT satisfies the hard uniqueness
constraint essentially perfectly (violation rate <1%) while also achieving high trajectory
diversity (>99.8% unique retained trajectories) — matching SQA on all three metrics (violation
rate, RMSE, diversity). This shows that constraint satisfaction and sample diversity are not in
fundamental tension for a classical sampler on this QUBO once its parameters are suitably tuned;
the difference from SQA is that SQA reaches this regime under its default annealing schedule,
whereas PT required this explicit temperature search. Together with folders 01 and 03, this gives
a three-way comparison that is central to the paper's argument:
- SQA: feasible **and** diverse, under its default schedule
- classical PT: feasible **and** diverse, once its replica temperature is selected via
  cross-validation
- hardware QA: diverse but largely infeasible (~95% violation)

# 02 — SA baseline (Simulated Annealing) and alpha/beta grid search

**Scripts:**
- `code/experiments/run_sa.py` (uses `code/src/solvers/sa.py`)
- `code/experiments/gridsearch.py` — two-stage (alpha, beta) grid search using SA as a fast proxy

**Data:** `data/observations.json`
**Run:**
```bash
uv run python experiments/run_sa.py                       # 30,000 reads, paper settings
uv run python experiments/gridsearch.py --solver sqa      # grid search (also usable with --solver sa)
```

**Results:**
- `results/sa_30k/` — RMSE = 0.0889, violation_rate = 0% (`traj.csv.gz` contains every decoded
  trajectory; gzip-compressed)
- `results/sa_gridsearch/` — two-stage grid search result; RMSE = 0.0699 with auto-selected (alpha, beta)
  (the same grid-search machinery, run as an SA proxy, also underlies the cross-validation,
  sensitivity, and bootstrap analyses in folder 07)

These are auxiliary classical references; the manuscript's main classical comparison points are
SQA (folder 01) and the Parallel Tempering baseline (folder 04).

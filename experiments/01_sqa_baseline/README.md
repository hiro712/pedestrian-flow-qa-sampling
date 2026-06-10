# 01 — SQA baseline (Simulated Quantum Annealing)

**Script:** `code/experiments/run_sqa.py` (uses `code/src/solvers/sqa.py`)
**Data:** `data/observations.json`
**Run:**
```bash
uv run python experiments/run_sqa.py            # 30,000 reads, paper settings (alpha=0.3, beta=0.55)
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa   # quick check
```

**Results:** `results/sqa_30k/`
- `results.json` — RMSE, violation_rate, loss, hyperparameters
- `*.png` — energy histograms, flow/proportion figures
- `sampleset.json.gz` / `traj.csv.gz` — every individual sample and decoded trajectory
  (gzip-compressed; decompress with `gunzip -k sampleset.json.gz`)

**Headline numbers (reported in the manuscript):** RMSE = 0.0777, violation_rate = 0%

This is the near-ideal classical reference sampler: it produces strictly feasible, diverse
trajectories and is the baseline against which both the QA hardware and the classical
Parallel Tempering baseline (folder 04) are compared.

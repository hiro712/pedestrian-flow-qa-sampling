# 03 — QA hardware baseline (D-Wave Advantage2, headline result)

**Script:** `code/experiments/run_qa.py` (uses `code/src/solvers/qa.py`, which wraps
`EmbeddingComposite(DWaveSampler(...))` and chunks `num_reads` into ≤5,000-read submissions)

**Data:** `data/observations.json`
**Hardware:** D-Wave `Advantage2_system1.6` — **note: this specific calibration has since been
retired by D-Wave**, so its `graph_id`/access timestamp could not be retroactively recovered. Folder
10 reruns the same pipeline on the currently available calibration as a same-chip corroboration.
**Run:** requires a `.env.local` with `DWAVE_SOLVER_NAME` / `DWAVE_API_TOKEN` (not included — see
top-level README "Reproducing the hardware experiments").
```bash
uv run python experiments/run_qa.py
```

**Results:** `results/qa_advantage2_30k/` (`results.json`, figures, plus
`sampleset.json.gz`/`traj.csv.gz` containing every individual sample and decoded trajectory,
gzip-compressed)

**Headline numbers (reported in the manuscript):** RMSE = 0.079481, violation_rate = 95.08%,
chain_break_fraction = 0

This is the central negative result of the paper: despite an RMSE numerically close to SQA's, the
hardware QA samples violate the one-hot uniqueness constraint in ~95% of trajectories and produce
qualitatively incoherent flow structure (Fig. "qa_F" in the manuscript) — motivating the
"sampler, not optimizer" framing and the diagnostic experiments in folders 05/06/09 and 10.

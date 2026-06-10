# 05 — Quantitative dynamic-range analysis & manual rescaling mitigation

**Hypothesis under test:** the manuscript notes that the QUBO's coefficients span a wide range
relative to the QPU's programmable `h`/`J` ranges, and that the device's automatic coefficient
compression (`auto_scale`) may itself be a source of the high constraint-violation rate seen in
the hardware-QA baseline (folder 03). This experiment makes that argument quantitative: it measures
the actual coefficient compression ratio, then tests whether disabling `auto_scale` and manually
rescaling coefficients to fill the device's range reduces the violation rate.

**Script:** `code/experiments/run_qa_autoscale.py`
- Computes the embedded-Ising coefficient table (`scale_info.json`: max/min |coeff|, their ratio,
  device h/J ranges, the linear rescaling factor applied)
- Re-submits with `auto_scale=False` and coefficients linearly rescaled to fill the device's
  programmable range
- Uses a **shared, cached minor-embedding** (`code/experiments/_embedding_cache.py` →
  `results/_embedding_cache/qubo_main_embedding.json`, 132 logical variables, chain lengths 6–17,
  mean ≈ 10.6, 1,397 physical qubits total) — the same embedding is reused by experiment 06,
  since spin-reversal transforms only flip coefficient signs and do not change the coupling graph

**Data:** `data/observations.json`
**Hardware:** D-Wave `Advantage2_system1` (current calibration; `graph_id = 01138bbada`)
**Run:** requires `.env.local` (see top-level README)
```bash
uv run python experiments/run_qa_autoscale.py
```

**Results:** `results/qa_autoscale_30k/` (`results.json`, `scale_info.json`, energy histograms,
plus `sampleset.json.gz`/`traj.csv.gz` with every individual sample and decoded trajectory,
gzip-compressed)

**Headline numbers:**
| | coefficient ratio (max/min) | violation_rate | RMSE |
|---|---|---|---|
| before rescaling | ≈ 139.8 | 95.54% (same-chip baseline, folder 10) | 0.0926 |
| after `auto_scale=False` + linear rescale | — | **56.51%** | **0.1064** |

Manual rescaling cuts the violation rate by ~39 percentage points relative to the matched
same-chip/same-calibration baseline (folder 10), confirming that coefficient compression is a real,
quantifiable contributor to the high baseline violation rate — though more than half the violations
persist, showing it is not the whole story (see folder 06 for a complementary mitigation that
targets a different error source, and how the two combine).

**Note on `num_reads` chunking:** the current `Advantage2_system1` calibration enforces a
per-submission ceiling of `num_reads <= 10000` (stricter than the `<= 5000` chunk size the
pipeline already used elsewhere); this script therefore submits in 6 chunks of 5,000 reads and
concatenates the results to reach R = 30,000. This kind of silent, calibration-dependent operational
difference is exactly why the manuscript reports chip/calibration identifiers explicitly.

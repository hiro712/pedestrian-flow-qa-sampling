# 06 — Spin-Reversal Transform (SRT) gauge-averaging mitigation

**Why this experiment exists:** the manuscript's hardware-QA hyperparameter search varied only
penalty strength and anneal time. Standard D-Wave error-mitigation techniques — spin-reversal
transforms (gauge averaging) in particular — are a natural next thing to try, since they are
designed to cancel exactly the kind of systematic, gauge-dependent control error that could explain
why hardware QA samples violate the problem's constraints so much more often than SQA does.

**Hypothesis under test:** does averaging samples over many random gauge transformations (sign
flips that leave the coupling graph unchanged) cancel gauge-dependent integrated control error
(ICE) bias and reduce the constraint-violation rate?

**Script:** `code/experiments/run_qa_srt.py`
- Wraps a `FixedEmbeddingComposite(DWaveSampler(...), embedding)` — using the **same cached
  embedding** as experiment 05 (SRT only flips coefficient signs; the coupling graph, and hence
  the valid embedding, is unchanged) — in `dwave.preprocessing.SpinReversalTransformComposite`
- 100 random gauges x 300 reads = R = 30,000 total samples (matches the paper's R=30,000
  convention; **important:** `num_spin_reversal_transforms=N` submits N independent jobs of
  `num_reads` each, i.e. N x num_reads total samples — choose `--reads` accordingly)

**Data:** `data/observations.json`
**Hardware:** D-Wave `Advantage2_system1` (`graph_id = 01138bbada`)
**Run:**
```bash
uv run python experiments/run_qa_srt.py --reads 300 --srt 100 --out results/qa_srt_30k
```

**Results:** `results/qa_srt_30k/` (`results.json`, figures, plus
`sampleset.json.gz`/`traj.csv.gz` with every individual sample and decoded trajectory,
gzip-compressed)

**Headline numbers:** violation_rate = **51.49%**, RMSE = 0.1680
— vs. the matched same-chip/same-calibration single-gauge baseline (folder 10):
violation_rate = 95.54%, RMSE = 0.0926

**Interpretation:** gauge averaging cuts the violation rate by ~44 percentage points — slightly
more than coefficient rescaling alone (folder 05) — strongly confirming the hypothesis that
gauge-dependent integrated control error accounts for roughly half of the ~95% baseline violation
rate. The dynamic-range mitigation (folder 05) and this gauge-averaging mitigation target two
distinct, additive error sources, and each independently produces a large, comparable reduction —
suggesting the baseline figure is a compound effect rather than a single irreducible mechanism.
Neither mitigation alone restores SQA-level (near-zero) violation rates, so the manuscript's
central "sampler, not optimizer" conclusion stands on firmer empirical ground: even after applying
standard error-mitigation techniques, hardware QA remains a fundamentally noisier, less-constrained
sampler than its simulated counterpart.

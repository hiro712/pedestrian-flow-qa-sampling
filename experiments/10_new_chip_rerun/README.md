# 10 — Rerun on a current hardware generation (corroboration & same-chip baseline)

**Why this experiment exists:** the solver calibration `Advantage2_system1.6` used for the
manuscript's headline QA results (folder 03) was **retired by D-Wave** after those results were
produced — only the newer calibration `Advantage2_system1` (`graph_id = 01138bbada`, a different
topology calibration) remains available. The manuscript's headline numbers are kept unchanged,
since they are a historical fact tied to the signs that were physically installed at the time the
study was conducted; this rerun instead serves two purposes:

1. **Corroboration** — checking that the qualitative finding (high constraint-violation rate,
   structurally different flow reconstruction relative to SQA) reproduces on a current,
   independently-calibrated chip, and is therefore not an artifact of one specific calibration.
2. **A fair same-chip, same-calibration baseline** — against which the dynamic-range mitigation
   (folder 05) and gauge-averaging mitigation (folder 06) experiments can be directly,
   apples-to-apples compared (the now-retired `Advantage2_system1.6` calibration can no longer
   provide this).

**Script:** `code/experiments/run_qa.py` (identical pipeline and hyperparameters as folder 03,
re-pointed at `Advantage2_system1` via `.env.local`)
**Data:** `data/observations.json`
**Hardware:** D-Wave `Advantage2_system1`, `graph_id = 01138bbada`, topology = Zephyr [12, 4]
**Run:** requires `.env.local` (see top-level README)
```bash
uv run python experiments/run_qa.py --out results/qa_advantage2_system1_30k
```

**Results:** `results/qa_advantage2_system1_30k/` (`results.json`, figures, plus
`sampleset.json.gz`/`traj.csv.gz` with every individual sample and decoded trajectory,
gzip-compressed)

**Headline numbers:** RMSE = 0.092575, violation_rate = 0.9554
(vs. the original-chip headline values RMSE = 0.079481, violation_rate = 0.9508 — folder 03)

**Additional finding:** the top-3 flow-corridor ranking also shifted between hardware generations
(original: 8-6, 6-7, 8-9 -> new: 6-7, 7-5, 1-3). Replacing the manuscript's historical numbers with
this rerun's would directly contradict its statement that signs were physically placed at the
(original) top-3 ranked locations — which is why the original numbers are retained in the main text
and this rerun is reported only as supplementary corroborating evidence.

**Note on `num_reads` chunking:** the current `Advantage2_system1` calibration enforces a stricter
per-submission ceiling (`num_reads <= 10000`) than the `<= 5000` chunk size the existing pipeline
assumed — one example of why explicit chip/calibration reporting matters for reproducibility:
operational details like submission limits can silently differ between calibrations of the
"same" device family.

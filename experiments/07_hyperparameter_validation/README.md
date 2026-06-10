# 07 — Hyper-parameter validation: LOO-CV, sensitivity, bootstrap CI

**Why this experiment exists:** the model's loss-function weights (alpha, beta) are chosen by
minimizing the same in-sample RMSE that the manuscript later reports as its headline fit metric —
a degree of circularity that needs to be addressed directly. This experiment separates "fit
quality on the data used for selection" from "predictive quality on held-out data," quantifies how
sensitive the results are to the chosen weights, and attaches uncertainty intervals to the
reported RMSE values.

**Script:** `code/experiments/m5_validation.py` (three analyses, all using an SA proxy for speed):
- (a) `loo_cv()` — leave-one-out cross-validation over all T=12 time blocks (grid search re-run
  per fold with the held-out block excluded)
- (b) `sensitivity_analysis()` — perturbs (alpha, beta) by +/-10% and measures the change in the
  reconstructed flow matrix F
- (c) `bootstrap_ci()` — 95% bootstrap confidence intervals (n_boot = 2000) for SQA/QA RMSE

**Data:** `data/observations.json`
**Run:**
```bash
uv run python experiments/m5_validation.py
```

**Results:** `results/m5_validation/m5_summary.json`

**Headline numbers:**
- LOO-CV: mean out-of-sample RMSE = 0.308 +/- 0.025 (12-fold)
- Sensitivity: mean |delta F| ~ 0.09-0.12 percentage points, max |delta F| = 1.0 pt
- Bootstrap 95% CI: SQA [0.07736, 0.07810], QA [0.07917, 0.07983] (non-overlapping, half-widths ~ 4e-4)

**Interpretation:** this separates "(alpha, beta) selected to minimize in-sample RMSE" from
"out-of-sample predictive quality" as two distinct, now-quantified numbers, resolving the
circularity concern raised by a purely in-sample fit metric; the tight, non-overlapping bootstrap
intervals confirm the small SQA-vs-QA RMSE gap is statistically real (which folder 08's
structure-aware divergence analysis then shows masks a much larger difference in
trajectory-level structure).

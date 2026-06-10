# 08 — KL / Jensen-Shannon divergence between SQA and QA flow distributions

**Why this experiment exists:** the manuscript argues visually (via figures) that SQA and hardware
QA produce reconstructed flow matrices with similar RMSE but substantially different underlying
structure. A visual contrast alone is not a strong basis for that claim — this experiment turns it
into a quantitative, reportable divergence measure between the two predicted flow distributions.

**Script:** `code/analysis/kl_divergence.py`
- No measured joint flow distribution exists (only row-normalized marginals are observable from
  headcounts), so the script directly compares the SQA and QA *predicted* flow matrices F as
  structure-aware distributions via KL and Jensen-Shannon divergence.

**Data:** the `results.json` outputs of experiments 01 (SQA) and 03 (QA hardware baseline)
**Run:**
```bash
uv run python analysis/kl_divergence.py
```

**Results:** `results/kl_divergence.json`

**Headline numbers:**
- D_KL(F_SQA || F_QA) = 0.383 nats
- D_KL(F_QA || F_SQA) = 1.055 nats
- Jensen-Shannon divergence = 0.098 nats (~14% of the maximum possible value, ln 2 ~ 0.693)

**Interpretation:** quantitatively substantiates "RMSE is close (folder 07's bootstrap CIs show
the ~0.018 gap is real but small) yet the structures are substantially different" — turning a
qualitative visual contrast (Fig. "sqa_F" vs. "qa_F" in the manuscript) into a reportable number.

**Supplementary corroboration:** a rerun of the same pipeline on a newer hardware generation
(folder 10) reproduced the same qualitative finding — high violation rate and structurally
different flow rankings — corroborating that this is a robust property of hardware QA on this
problem rather than an artifact of one specific chip calibration.

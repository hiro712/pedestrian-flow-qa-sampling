# 09 — Effective inverse temperature (beta_eff) and total-variation distance

**Why this experiment exists:** the manuscript's Discussion describes SQA as operating at a much
lower effective temperature than hardware QA — a claim used to explain why the two samplers explore
the solution space so differently. This experiment makes that comparison quantitative by fitting an
effective inverse temperature to each sampler's output on a tractable sub-problem.

**Script:** `code/analysis/boltzmann_subproblem.py`
- Reduces the problem to one time slice (2^11 = 2,048 enumerable states), MLE-fits an effective
  inverse temperature beta_eff to the empirical sample distribution (following the approach of
  Benedetti et al.), and computes the total-variation (TV) distance to the resulting Boltzmann
  distribution. Applied separately to the SQA and hardware-QA sample sets.

**Data / chip attribution:** both subproblem fits below were computed on samples from the **same
dataset and chip generation** as the manuscript's headline QA results (`results/qa_advantage2_30k/`,
solver `Advantage2_system1.6`, since retired by D-Wave). This analysis was deliberately **not**
rerun on the newer hardware generation used in folders 05/06/10: doing so would mix chip
generations within a single reported quantity, undermining the comparison it is meant to support.

**Run:**
```bash
uv run python analysis/boltzmann_subproblem.py
```

**Results:**
- `results/sqa_30k_subproblem/boltzmann_subproblem_t6.{json,png}`
- `results/qa_advantage2_30k_subproblem/boltzmann_subproblem_t6.{json,png}`

**Headline numbers:**
| | beta_eff | T_eff | TV distance |
|---|---|---|---|
| SQA | 49.27 | ~ 0.020 | 0.256 |
| hardware QA | 0.041 | ~ 24.6 | 0.340 |

**Interpretation:** SQA operates at an effective temperature roughly **1,200x lower** than the
hardware annealer — a quantitative anchor for the qualitative "effective-temperature gap" argument,
now reflected in the manuscript's Discussion section alongside the existing description of the
QA-vs-SQA failure-mode asymmetry (broad-but-infeasible vs. feasible-and-diverse).

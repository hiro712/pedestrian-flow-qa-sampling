# code/ folder structure notes

Last updated: 2026-06-10

This file describes the structure of `public_release/code/`. For the overall repository layout
(the relationship to `../data/`, `../experiments/`), see the [top-level README](../README.md).

---

## Folder tree

```
code/
├── src/                        # core library (importable)
│   ├── __init__.py
│   ├── graph.py                # get_distances() — graph and distance matrix
│   ├── transition.py           # build_transition_P() — transition probability matrix
│   ├── qubo.py                 # build_qubo() — build the QUBO dict
│   ├── trajectory.py           # decode_traj(), compute_violation_rate(), save_results()
│   ├── flow.py                 # reconstruct_proportions(), build_flow_matrix()
│   ├── metrics.py              # rmse(), squared_loss()
│   └── solvers/
│       ├── __init__.py         # get_solver(type) — factory function
│       ├── base.py             # SolverBase, Qubo, SampleConfig type definitions
│       ├── sa.py                # SASolver (OpenJij SA)
│       ├── sqa.py              # SQASolver (OpenJij SQA)
│       ├── qa.py               # QASolver (D-Wave hardware)
│       └── parallel_tempering.py # PTSolver + run_all_replicas() (classical baseline, M4/FB1)
│
├── experiments/                # runnable scripts (uv run python experiments/xxx.py)
│   ├── _pipeline.py            # run_experiment(), run_gridsearch() — common pipeline
│   ├── _embedding_cache.py     # minor-embedding cache
│   ├── run_sqa.py              # SQA experiment (reproduces the manuscript: alpha=0.3, beta=0.55, reads=30k)
│   ├── run_sa.py               # SA experiment
│   ├── run_qa.py               # D-Wave hardware experiment
│   ├── run_qa_autoscale.py     # auto_scale=False + manual coefficient scaling (M1) / rerun on a new chip (10)
│   ├── run_qa_srt.py           # spin-reversal transform / gauge averaging (M2)
│   ├── run_pt.py               # Parallel Tempering experiment, replica temperature selected via 10-fold CV (M4/FB1)
│   ├── m5_validation.py        # LOO-CV, sensitivity analysis, bootstrap CI (M5)
│   └── gridsearch.py           # alpha/beta grid search -> final run
│
├── analysis/                   # figure-generation / diagnostic scripts (take results.json)
│   ├── plot_proportions.py     # p_true vs p_prime heatmap
│   ├── plot_flow_matrix.py     # heatmap of the flow matrix F
│   ├── plot_edge_flow.py       # visualize edge flow on the graph
│   ├── boltzmann_subproblem.py # beta_eff fit and TV distance (reduced sub-problem, M11)
│   └── kl_divergence.py        # KL/JS divergence between the SQA and QA flow matrices (M10)
│
├── figures/                    # scripts that generate the manuscript's figures
│   └── out/                    # destination for generated figures
│
├── gridsearch/, distance/, predict/, solve/   # auxiliary modules
│
├── results/                    # output destination when scripts are run (git-ignored; regenerated on rerun)
│
├── data/
│   └── observations.json       # input data (observed headcounts, 12 time steps x 10 zones)
│                                # identical to ../data/observations.json
│
├── pyproject.toml              # uv dependency configuration
└── STRUCTURE.md                # this file
```

The curated experiment results (annotated with the value reported in the manuscript) live not
under `code/results/` but under `experiments/<NN_experiment_name>/results/` at the repository root
(see the table below). To rerun an experiment that uses D-Wave hardware, set `DWAVE_API_TOKEN` etc.
in `code/.env.local` (git-ignored).

---

## Processing flow

```
data/observations.json
        ↓
src/graph.py          →  distance matrix D (11x11, median-normalized)
        ↓
src/transition.py     →  transition probability matrix P (11x11)  <- controlled by alpha, beta
        ↓
src/qubo.py           →  QUBO dict Q              <- controlled by 5 lambda coefficients
        ↓
src/solvers/          →  SampleSet (SA / SQA / QA / PT)
        ↓
src/trajectory.py     →  trajectory list (num_reads of them)
        ↓
src/flow.py           →  p' (T x N), F (11x11)
        ↓
src/metrics.py        →  RMSE, violation_rate
        ↓
results/<name>/       →  results.json, traj.csv, sampleset.json, *.png
```

---

## How to run the experiments

```bash
# Smoke test (100 reads)
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
uv run python experiments/run_sa.py  --reads 100 --out results/test_sa

# Reproduce the manuscript (30000 reads)
uv run python experiments/run_sqa.py
uv run python experiments/run_sa.py

# D-Wave hardware (requires a token set in code/.env.local)
uv run python experiments/run_qa.py

# alpha/beta grid search
uv run python experiments/gridsearch.py --solver sqa

# Generate figures (example using the curated results)
uv run python analysis/plot_proportions.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json
uv run python analysis/plot_flow_matrix.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json
uv run python analysis/plot_edge_flow.py   ../experiments/01_sqa_baseline/results/sqa_30k/results.json
```

---

## QUBO parameters (values used in the manuscript)

| Parameter | Value | Meaning |
|---|---|---|
| `lambda_onehot` | 13.0 | One-hot constraint (hard) |
| `lambda_P` | 5.0 | Adherence to the transition probabilities |
| `lambda_div` | 1.0 | Visit dispersion |
| `lambda_entry` | 2.0 | Control of entry/exit through the outside node |
| `lambda_move` | 0.5 | Smoothing of internal moves |
| `alpha` | 0.3 | Distance-decay coefficient |
| `beta` | 0.55 | Destination-popularity exponent |

---

## Summary of existing results

The curated results for the manuscript and supplementary checks are stored not under `code/` but
under `experiments/<NN_experiment_name>/results/` at the repository root.

| Directory (under `../experiments/`) | Solver | RMSE | violation_rate | Notes |
|---|---|---|---|---|
| `01_sqa_baseline/results/sqa_30k/` | SQA | 0.0777 | 0% | **Value reported in the manuscript** |
| `02_sa_baseline/results/sa_30k/` | SA | 0.0889 | 0% | Reference value |
| `02_sa_baseline/results/sa_gridsearch/` | SA + two-stage search | 0.0699 | 0% | alpha/beta auto-selected |
| `03_qa_hardware_baseline/results/qa_advantage2_30k/` | D-Wave Advantage2_system1.6 | 0.0795 | 95.1% | **Value reported in the manuscript** (1.6 is retired) |
| `04_classical_pt_baseline/results/pt_30k/` | Parallel Tempering (temperature selected via CV) | 0.0692 | 0.6% | **Value reported in the manuscript** (replica temperature selected via 10-fold CV, M4 FB1) |
| `05_dynamic_range_analysis/results/qa_autoscale_30k/` | D-Wave Advantage2_system1 (manual scaling) | 0.1064 | 56.5% | Supplementary check (M1) |
| `06_gauge_averaging_srt/results/qa_srt_30k/` | D-Wave Advantage2_system1 (SRT, 100 gauges) | 0.1680 | 51.5% | Supplementary check (M2) |
| `07_hyperparameter_validation/results/m5_validation/` | — | LOO-CV 0.308±0.026 | — | Sensitivity analysis, bootstrap CI (M5) |
| `08_kl_divergence_analysis/results/kl_divergence.json` | — | — | — | KL/JS divergence between SQA and QA (M10) |
| `09_effective_temperature_analysis/results/` | SQA / QA | — | — | beta_eff, TV distance (M11) |
| `10_new_chip_rerun/results/qa_advantage2_system1_30k/` | D-Wave Advantage2_system1 | 0.0926 | 95.5% | Rerun on a newer chip (M10 supplement) |

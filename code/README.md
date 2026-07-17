# Pedestrian Flow Prediction Using QA Sampling

Experiment code for pedestrian flow prediction and sign placement using QA sampling.

This directory is the `code/` of the `public_release/` repository. For the overall repository
layout, and how the results contained here map onto the values reported in the manuscript, see the
[top-level README](../README.md).

---

## Setup

```bash
# Run from this directory (code/)
cd code/

# Install dependencies (first time only)
uv sync

# Smoke test
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
```

---

## Directory layout

```
code/
├── src/            core library (graph, transition, qubo, trajectory, flow, metrics, solvers/)
├── experiments/    experiment scripts
├── figures/        scripts that generate the manuscript's figures
├── analysis/       data-inspection / diagnostic scripts
├── gridsearch/, distance/, predict/, solve/  auxiliary modules
├── data/           input data (observations.json; identical to ../data/observations.json)
├── results/        output destination when scripts are run (git-ignored; regenerated on rerun)
└── STRUCTURE.md    detailed reference for the code structure
```

The curated results for each experiment (annotated with the value reported in the manuscript) are
not stored here, but under `../experiments/<NN_experiment_name>/results/` at the repository root.
See [STRUCTURE.md](STRUCTURE.md) and the [top-level README](../README.md) for details.

---

## Running the experiments

### SQA (Simulated Quantum Annealing) — the manuscript's headline result

```bash
# Reproduce the manuscript (30000 reads, roughly a few minutes to tens of minutes)
uv run python experiments/run_sqa.py

# Smoke test (100 reads, a few seconds)
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
```

Results are saved to `results/sqa_30k/` (curated copy of the manuscript value:
[`../experiments/01_sqa_baseline/results/sqa_30k/`](../experiments/01_sqa_baseline/results/sqa_30k/)). Value reported in the manuscript: **RMSE = 0.077708**

### SA (Simulated Annealing)

```bash
uv run python experiments/run_sa.py
uv run python experiments/run_sa.py --reads 100 --out results/test_sa
```

Results are saved to `results/sa_30k/` (curated copy: [`../experiments/02_sa_baseline/results/sa_30k/`](../experiments/02_sa_baseline/results/sa_30k/)). Reference value in the manuscript: RMSE = 0.0889

### D-Wave hardware (QA)

Prerequisite: set credentials in `.env.local` (see "D-Wave configuration" below).

```bash
# Check available solvers
uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env.local')
from dwave.cloud import Client
print([s.id for s in Client.from_config().get_solvers()])
"

# Run (30000 reads)
uv run python experiments/run_qa.py

# To change the annealing time
uv run python experiments/run_qa.py --annealing-time 100

# To change the output destination
uv run python experiments/run_qa.py --out results/qa_srt_test
```

### alpha/beta grid search

```bash
# Search with the SA backend (recommended: fast)
uv run python experiments/gridsearch.py --solver sa --reads 100

# Search with the SQA backend
uv run python experiments/gridsearch.py --solver sqa --reads 100 --final-reads 30000
```

The grid search can be resumed (cached in `history_stage1.csv` / `history_stage2.csv`).

---

## Generating the manuscript's figures

### Generate all figures in one batch

```bash
# Generate all figures from the SQA and QA results (saved to figures/out/)
uv run python figures/generate_all.py

# Specify the output destination
uv run python figures/generate_all.py --out /path/to/paper/figures/

# Specify which results.json to use (example: using the curated results directly)
uv run python figures/generate_all.py \
    --sqa ../experiments/01_sqa_baseline/results/sqa_30k/results.json \
    --qa  ../experiments/03_qa_hardware_baseline/results/qa_advantage2_30k/results.json \
    --out figures/out/
```

### Generate individually

```bash
# Venue graph (Fig 1)
uv run python figures/fig_graph.py --out figures/out/

# Observed data (zone totals + time-series heatmap)
uv run python figures/fig_data.py --out figures/out/

# p_true vs p_prime heatmap (Fig 2)
uv run python figures/fig_proportions.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/

# Flow matrix F (Fig 3)
uv run python figures/fig_flow_matrix.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/

# Edge flow ratio graph (Fig 4)
uv run python figures/fig_edge_flow.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/
uv run python figures/fig_edge_flow.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --top 5  # highlight the top 5
```

---

## Inspecting experiment results

Each experiment's output directory contains the following:

| File | Contents |
|---|---|
| `results.json` | RMSE, violation_rate, P, F, p_true, p_prime, etc. |
| `sampleset.json` | The full sampleset (SQA: ~300MB, QA: ~60MB) |
| `traj.csv` | All trajectories (rows = samples, columns = time steps) |
| `energy_histogram_hw.png` | Hardware-reported energy distribution |
| `energy_histogram_overlay.png` | Raw vs Fixed (recomputed QUBO) energy comparison |
| `energy_histogram_fixed.png` | Fixed-only energy distribution |

```bash
# To check just RMSE and violation_rate (example for a result you ran yourself)
python -c "
import json; r = json.load(open('results/sqa_30k/results.json'))
print(f'RMSE={r[\"rmse\"]:.6f}  violation={r[\"violation_rate\"]:.4f}')
"

# To check the curated (manuscript-reported) results directly
python -c "
import json; r = json.load(open('../experiments/01_sqa_baseline/results/sqa_30k/results.json'))
print(f'RMSE={r[\"rmse\"]:.6f}  violation={r[\"violation_rate\"]:.4f}')
"
```

---

## D-Wave configuration

Set the following in `.env.local` (git-ignored):

```
DWAVE_SOLVER_NAME=Advantage2_system2.3
DWAVE_API_TOKEN=your-token-here
```

**Available solvers** change over time. Check with the command above before running.

Solvers used previously:
- `Advantage2_system1.6` — retired (the manuscript value was obtained on this solver)
- `Advantage2_system1.11` — possibly retired

---

## QUBO parameters

Values used in the manuscript. To change them, edit the constants at the top of each
`experiments/run_*.py` script.

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.3 | Distance-decay coefficient |
| `beta` | 0.55 | Destination-popularity exponent |
| `lambda_onehot` | 13.0 | One-hot constraint (hard constraint) |
| `lambda_P` | 5.0 | Adherence to the transition probabilities (soft) |
| `lambda_div` | 1.0 | Visit dispersion |
| `lambda_entry` | 2.0 | Control of entry/exit through the outside node |
| `lambda_move` | 0.5 | Smoothing of internal moves |

---

## Existing experiment results (no rerun needed)

The curated results reported in the manuscript are not stored under `code/`, but under
`experiments/<NN_experiment_name>/results/` at the repository root.

| Directory (under `../experiments/`) | Solver | RMSE | violation_rate | Notes |
|---|---|---|---|---|
| `01_sqa_baseline/results/sqa_30k/` | SQA | 0.077708 | 0.0% | **Value reported in the manuscript** |
| `03_qa_hardware_baseline/results/qa_advantage2_30k/` | D-Wave Advantage2_system1.6 | 0.079481 | 95.1% | **Value reported in the manuscript** (solver retired) |
| `02_sa_baseline/results/sa_30k/` | SA | 0.088900 | 0.0% | Reference value |
| `02_sa_baseline/results/sa_gridsearch/` | SA + two-stage search | 0.069900 | 0.0% | alpha/beta auto-selected |
| `04_classical_pt_baseline/results/pt_30k/` | Parallel Tempering (temperature selected via CV) | 0.069184 | 0.6% | **Value reported in the manuscript** (replica temperature selected via 10-fold CV; see `cv_summary.json`) |
| `05_dynamic_range_analysis/results/qa_autoscale_30k/` | D-Wave Advantage2_system1 (manual scaling) | 0.106361 | 56.5% | Supplementary check (M1) |
| `06_gauge_averaging_srt/results/qa_srt_30k/` | D-Wave Advantage2_system1 (SRT, 100 gauges) | 0.167999 | 51.5% | Supplementary check (M2) |
| `10_new_chip_rerun/results/qa_advantage2_system1_30k/` | D-Wave Advantage2_system1 | 0.092575 | 95.5% | Rerun on a newer chip (M10 supplement) |

Each experiment folder's README describes, in detail, the corresponding claim in the manuscript,
the script that produced it, and the interpretation.

# Pedestrian Flow Prediction & Sign Placement — Code, Data, and Experiment Results

This repository accompanies the manuscript **"Prediction-based Sign Placement Using QA Sampling"**
(Scientific Reports submission `554d4ca8-d948-47c3-8425-8ca912912c3c`). It contains the full
analysis pipeline, the input observational data, and the curated results of every experiment
referenced in the manuscript — both the headline results reported in the main text and a set of
supplementary diagnostic experiments that probe *why* hardware quantum annealing behaves the way
it does on this problem.

## Layout

```
public_release/
├── README.md            <- you are here
├── code/                <- the runnable analysis pipeline (importable library + run scripts)
│   ├── src/             <-   core library: graph/distance, transition model, QUBO construction,
│   │                         trajectory decoding, flow reconstruction, metrics, solvers
│   │                         (SA, SQA, hardware QA, Parallel Tempering)
│   ├── experiments/     <-   "uv run python experiments/run_xxx.py" entry points
│   ├── analysis/        <-   figure-/statistic-generation scripts that consume results.json
│   └── STRUCTURE.md     <-   detailed module-by-module map of code/ (data flow diagram, etc.)
├── data/
│   ├── README.md           <-   read this first: exactly what the observed values are
│   └── observations.json   <- input: 12 time bins x 10 zones, per-zone occupancies C[t][i]
│                                 (accumulated net boundary crossings; see data/README.md)
└── experiments/         <- ** start here ** — one numbered folder per experiment:
    │                         each README explains the scientific question the experiment answers,
    │                         which script + data produced it, and what it found; each `results/`
    │                         subfolder is a curated copy of the corresponding run's output
    │                         (see note below)
    ├── 01_sqa_baseline/                  Simulated Quantum Annealing — main classical reference
    ├── 02_sa_baseline/                   Simulated Annealing baseline + hyperparameter grid search
    ├── 03_qa_hardware_baseline/          D-Wave hardware QA — the manuscript's headline QA result
    ├── 04_classical_pt_baseline/         Parallel Tempering — matched classical-sampler comparison
    ├── 05_dynamic_range_analysis/        does coefficient compression explain QA's high violation rate?
    ├── 06_gauge_averaging_srt/           does gauge-dependent control error explain the rest of it?
    ├── 07_hyperparameter_validation/     out-of-sample validation, sensitivity, bootstrap CIs
    ├── 08_kl_divergence_analysis/        quantifying how different SQA's and QA's flow structures are
    ├── 09_effective_temperature_analysis/ quantifying the SQA-vs-QA effective-temperature gap
    └── 10_new_chip_rerun/                corroboration rerun on a current hardware generation
```

The numbering follows the logical order in which the experiments are introduced and motivated in
the manuscript: the baselines first (01–04), then a series of diagnostic experiments (05–09) that
dig into *why* hardware QA's results look the way they do relative to SQA, and finally a
hardware-generation corroboration rerun (10). It is **not** strictly chronological.

## How to reproduce an experiment

1. `cd code/`
2. Install dependencies: `uv sync` (requires [uv](https://docs.astral.sh/uv/); see `pyproject.toml`
   for the pinned versions — OpenJij for SA/SQA, `dwave-ocean-sdk` for hardware QA, numpy/scipy etc.)
3. Open the numbered folder for the experiment you want to reproduce (e.g.
   `experiments/06_gauge_averaging_srt/README.md`) — it tells you exactly which script to run
   and with which arguments.
4. Run the script from inside `code/`, e.g.:
   ```bash
   uv run python experiments/run_sqa.py
   ```
   Output is written to `code/results/<name>/`; compare it against the curated reference copy in
   the corresponding numbered folder's `results/` subdirectory.

### Reproducing the hardware (D-Wave QA) experiments

Experiments 03, 05, 06, and 10 require access to D-Wave hardware via the Ocean SDK. Create a file
`code/.env.local` (gitignored, **never commit this file**) with:
```
DWAVE_SOLVER_NAME="<solver name, e.g. Advantage2_system1>"
DWAVE_API_TOKEN="<your D-Wave Leap API token>"
```
**Note on chip generations:** the calibration `Advantage2_system1.6` used for the manuscript's
original headline QA results (folder 03) has since been retired by D-Wave. Folder 10 documents
why the manuscript's headline numbers were nonetheless kept unchanged, and how the newer
`Advantage2_system1` calibration was instead used as a fair same-chip, same-calibration baseline
for the diagnostic mitigation experiments in folders 05 and 06. This also surfaced a concrete
example of why explicit chip/calibration reporting matters for reproducibility: the two
calibrations enforce different per-submission `num_reads` limits (see folder 05's README).

## Raw per-sample data

In addition to the curated `results.json` (RMSE/violation-rate/etc.), `scale_info.json` where
applicable, and figures, each numbered experiment folder's `results/` subdirectory also includes
the raw per-sample outputs — `sampleset.json.gz` and `traj.csv.gz`, every individual annealer
sample and decoded trajectory, gzip-compressed (decompress with `gunzip -k <file>.gz`). This is
everything needed to independently verify or re-analyze the reported numbers.

## License

The source code in `code/` is licensed under the [MIT License](LICENSE). The contents of `data/`
and `experiments/` (observational data, curated results, and figures) are licensed under
[CC BY 4.0](LICENSE-DATA).

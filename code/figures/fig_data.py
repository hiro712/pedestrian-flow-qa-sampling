"""
Fig 1 supplement: bar chart of the observed occupancies C[t][i] summed over all
time bins, per zone (for data inspection and the manuscript).

Usage:
    uv run python figures/fig_data.py
    uv run python figures/fig_data.py --data data/observations.json --out figures/out/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_data(data_path: Path, out_dir: Path) -> None:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    arr = np.array(data, dtype=float)  # (T, N)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    T, N = arr.shape
    zone_totals = arr.sum(axis=0)
    x = np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: bar chart of zone totals ---
    axes[0].bar(x, zone_totals, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Zone index")
    axes[0].set_ylabel("Total observed (sum over all time steps)")
    axes[0].set_title("Total pedestrian count per zone")
    axes[0].set_xticks(x)

    # --- Right: time-series heatmap ---
    im = axes[1].imshow(arr.T, aspect="auto", origin="lower", cmap="Blues")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Zone index")
    axes[1].set_title("Observed counts $C$ (time × zone)")
    axes[1].set_yticks(np.arange(N))
    axes[1].set_yticklabels([str(i) for i in range(1, N + 1)])
    axes[1].set_xticks(np.arange(T))
    axes[1].set_xticklabels([str(t + 1) for t in range(T)], fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_data.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Shape: T={T}, N={N}")
    for i, v in enumerate(zone_totals, 1):
        print(f"  Zone {i:2d}: {int(v)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/observations.json", help="path to observations.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1] / args.data
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "out"
    plot_data(data_path, out_dir)


if __name__ == "__main__":
    main()

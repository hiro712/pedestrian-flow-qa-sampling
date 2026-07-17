"""
Generate a figure comparing heatmaps of p_true and p_prime.

Usage:
    uv run python analysis/plot_proportions.py results/sqa_30k/results.json
    uv run python analysis/plot_proportions.py results/sqa_30k/results.json --out figures/
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str, help="path to results.json")
    parser.add_argument("--out", type=str, default=None, help="output directory (default: same location as results.json)")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    p_true = np.array(data["p_true"])
    p_prime = np.array(data["p_prime"])
    diff = p_true - p_prime
    rmse_val = data.get("rmse", float("nan"))
    solver = data.get("solver", "?")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = max(p_true.max(), p_prime.max())

    im0 = axes[0].imshow(p_true.T, aspect="auto", vmin=0, vmax=vmax, cmap="Blues")
    axes[0].set_title("p_true (observed)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Zone")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(p_prime.T, aspect="auto", vmin=0, vmax=vmax, cmap="Blues")
    axes[1].set_title(f"p_prime ({solver})")
    axes[1].set_xlabel("Time step")
    plt.colorbar(im1, ax=axes[1])

    lim = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff.T, aspect="auto", vmin=-lim, vmax=lim, cmap="RdBu_r")
    axes[2].set_title(f"|diff|  RMSE={rmse_val:.6f}")
    axes[2].set_xlabel("Time step")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out_path = out_dir / "fig_proportions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

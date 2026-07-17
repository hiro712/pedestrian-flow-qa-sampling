"""
Fig 2: three-panel heatmap of p_true vs p_prime (for the manuscript).

Usage:
    uv run python figures/fig_proportions.py results/sqa_30k/results.json
    uv run python figures/fig_proportions.py results/sqa_30k/results.json --out figures/out/
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


def plot_proportions(results_path: Path, out_dir: Path) -> None:
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)

    p_true = np.array(res["p_true"])   # (T, N)
    p_prime = np.array(res["p_prime"]) # (T, N)
    diff = p_prime - p_true
    rmse_val = res.get("rmse", float("nan"))
    solver = res.get("solver", "?")

    T, N = p_true.shape
    zone_labels = [str(i) for i in range(1, N + 1)]

    def _imshow(ax, M, title, cmap="Blues", vmin=None, vmax=None, center=False):
        if center:
            lim = max(abs(M.min()), abs(M.max()))
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Zone")
        ax.set_title(title)
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(zone_labels)
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([str(t + 1) for t in range(T)], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vmax = max(p_true.max(), p_prime.max())
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    _imshow(axes[0], p_true,  "Observed proportion $p$",              vmin=0, vmax=vmax)
    _imshow(axes[1], p_prime, f"Reconstructed $p'$ ({solver})",       vmin=0, vmax=vmax)
    _imshow(axes[2], diff,    f"Difference $p'-p$  (RMSE={rmse_val:.6f})", center=True)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_proportions.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="path to results.json")
    parser.add_argument("--out", default=None, help="output directory (default: same location as results.json)")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent
    plot_proportions(results_path, out_dir)


if __name__ == "__main__":
    main()

"""
Fig 3: heatmap of the flow matrix F (for the manuscript).

Usage:
    uv run python figures/fig_flow_matrix.py results/sqa_30k/results.json
    uv run python figures/fig_flow_matrix.py results/sqa_30k/results.json --out figures/out/
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


def plot_flow_matrix(results_path: Path, out_dir: Path) -> None:
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)

    F = np.array(res["F"])   # (N+1, N+1)
    solver = res.get("solver", "?")
    n = F.shape[0]
    labels = ["out"] + [str(i) for i in range(1, n)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(F, origin="lower", aspect="equal", cmap="YlOrRd", vmin=0)
    ax.set_xlabel("Destination node $j$")
    ax.set_ylabel("Source node $i$")
    ax.set_title(f"Row-normalized flow matrix $F$ ({solver})")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_flow_matrix.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="path to results.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent
    plot_flow_matrix(results_path, out_dir)


if __name__ == "__main__":
    main()

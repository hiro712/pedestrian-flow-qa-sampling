"""
Generate a heatmap of the flow matrix F.

Usage:
    uv run python analysis/plot_flow_matrix.py results/sqa_30k/results.json
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
    parser.add_argument("results_json", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    F = np.array(data["F"])
    solver = data.get("solver", "?")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(F, cmap="YlOrRd", vmin=0)
    ax.set_title(f"Flow matrix F ({solver})")
    ax.set_xlabel("Destination zone (0=outside)")
    ax.set_ylabel("Source zone (0=outside)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out_path = out_dir / "fig_flow_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

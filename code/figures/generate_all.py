"""
Script to generate all figures for the manuscript in one batch.

Usage:
    uv run python figures/generate_all.py
    uv run python figures/generate_all.py --sqa results/sqa_30k/results.json --qa results/qa_advantage2_30k/results.json
    uv run python figures/generate_all.py --out figures/out/

Generated figures:
    fig_graph.png            ... venue graph structure
    fig_data.png             ... observed data (zone totals + time-series heatmap)
    fig_proportions_sqa.png  ... p_true vs p_prime (SQA)
    fig_proportions_qa.png   ... p_true vs p_prime (QA)
    fig_flow_matrix_sqa.png  ... flow matrix F (SQA)
    fig_flow_matrix_qa.png   ... flow matrix F (QA)
    fig_edge_flow_sqa.png    ... edge flow (SQA)
    fig_edge_flow_qa.png     ... edge flow (QA)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.fig_graph import plot_graph
from figures.fig_data import plot_data
from figures.fig_proportions import plot_proportions
from figures.fig_flow_matrix import plot_flow_matrix
from figures.fig_edge_flow import plot_edge_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqa", default="results/sqa_30k/results.json",
                        help="path to SQA's results.json")
    parser.add_argument("--qa", default="results/qa_advantage2_30k/results.json",
                        help="path to QA's results.json")
    parser.add_argument("--data", default="data/observations.json")
    parser.add_argument("--out", default="figures/out", help="output directory")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== fig_graph ===")
    plot_graph(out_dir)

    print("\n=== fig_data ===")
    plot_data(root / args.data, out_dir)

    def _run_result_figs(results_json_str: str, suffix: str) -> None:
        rpath = Path(results_json_str) if Path(results_json_str).is_absolute() \
                else root / results_json_str
        if not rpath.exists():
            print(f"  [skip] {rpath} not found")
            return

        print(f"\n=== fig_proportions ({suffix}) ===")
        plot_proportions(rpath, out_dir)
        # Append the suffix to the filename
        (out_dir / "fig_proportions.png").rename(out_dir / f"fig_proportions_{suffix}.png")

        print(f"\n=== fig_flow_matrix ({suffix}) ===")
        plot_flow_matrix(rpath, out_dir)
        (out_dir / "fig_flow_matrix.png").rename(out_dir / f"fig_flow_matrix_{suffix}.png")

        print(f"\n=== fig_edge_flow ({suffix}) ===")
        plot_edge_flow(rpath, out_dir, top=args.top)
        (out_dir / "fig_edge_flow.png").rename(out_dir / f"fig_edge_flow_{suffix}.png")

    _run_result_figs(args.sqa, "sqa")
    _run_result_figs(args.qa, "qa")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()

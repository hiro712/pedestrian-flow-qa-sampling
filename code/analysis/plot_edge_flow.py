"""
Visualize edge flow ratios on the graph.

Usage:
    uv run python analysis/plot_edge_flow.py results/sqa_30k/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import EDGES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--top", type=int, default=3, help="number of top flows to highlight")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    F = np.array(data["F"])
    solver = data.get("solver", "?")

    G = nx.Graph()
    G.add_weighted_edges_from(EDGES)
    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    # Flow between internal zones (excluding the outside node 0)
    n = F.shape[0]
    edge_flows: dict[tuple[int, int], float] = {}
    total_flow = 0.0
    for i in range(1, n):
        for j in range(1, n):
            if i != j and G.has_edge(i, j):
                flow = F[i, j] + F[j, i]
                edge_flows[(min(i, j), max(i, j))] = edge_flows.get((min(i, j), max(i, j)), 0.0) + flow / 2
                total_flow += flow / 2

    top_edges = sorted(edge_flows.items(), key=lambda x: -x[1])[: args.top]
    top_set = {e for e, _ in top_edges}

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)

    for (u, v), flow in edge_flows.items():
        ratio = flow / total_flow if total_flow > 0 else 0
        width = 1 + 10 * ratio
        color = "red" if (u, v) in top_set else "gray"
        alpha = 0.8 if (u, v) in top_set else 0.3
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, alpha=alpha, ax=ax)
        mid = (np.array(pos[u]) + np.array(pos[v])) / 2
        ax.annotate(f"{ratio*100:.1f}%", xy=mid, fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(f"Edge flow ratios ({solver})  top-{args.top} in red")
    ax.axis("off")
    plt.tight_layout()

    out_path = out_dir / "fig_edge_flow.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

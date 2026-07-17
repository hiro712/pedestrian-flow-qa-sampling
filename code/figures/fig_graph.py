"""
Fig 1: visualization of the venue graph (nodes, edges, distance weights).

Usage:
    uv run python figures/fig_graph.py
    uv run python figures/fig_graph.py --out figures/out/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import EDGES


def plot_graph(out_dir: Path) -> None:
    G = nx.Graph()
    G.add_weighted_edges_from(EDGES)

    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Nodes (outside = 0 in a different color)
    internal = [n for n in G.nodes() if n != 0]
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="salmon",
                           node_size=1200, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=internal, node_color="lightblue",
                           node_size=1200, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=16, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6, ax=ax)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        ax=ax,
    )

    ax.set_title("Venue graph (node 0 = outside, nodes 1–10 = zones)")
    ax.axis("off")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "out"
    plot_graph(out_dir)


if __name__ == "__main__":
    main()

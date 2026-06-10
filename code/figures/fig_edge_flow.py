"""
Fig 4: グラフ上にエッジフロー比率を可視化（論文用）。

使い方:
    uv run python figures/fig_edge_flow.py results/sqa_30k/results.json
    uv run python figures/fig_edge_flow.py results/sqa_30k/results.json --top 3 --out figures/out/
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


def plot_edge_flow(results_path: Path, out_dir: Path, top: int = 3) -> None:
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)

    F = np.array(res["F"])   # (N+1, N+1)
    solver = res.get("solver", "?")

    G = nx.Graph()
    G.add_weighted_edges_from(EDGES)
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # エッジごとのフロー比率（全遷移に対する割合）
    total_flow = F.sum()
    edge_flow_ratio: dict[tuple[int, int], float] = {}
    for u, v, _ in EDGES:
        ratio = (F[u, v] + F[v, u]) / total_flow if total_flow > 0 else 0.0
        edge_flow_ratio[(u, v)] = ratio

    # コンソールに表示
    print(f"Total flow: {total_flow:,.0f}")
    for rank, ((u, v), ratio) in enumerate(
        sorted(edge_flow_ratio.items(), key=lambda x: -x[1]), 1
    ):
        marker = " <-- sign" if rank <= top else ""
        print(f"  {rank:2d}. ({u:2d}-{v:2d}): {ratio*100:5.2f}%{marker}")

    top_edges = {e for e, _ in sorted(edge_flow_ratio.items(), key=lambda x: -x[1])[:top]}
    max_ratio = max(edge_flow_ratio.values()) if edge_flow_ratio else 1.0

    fig, ax = plt.subplots(figsize=(14, 10))

    # ノード
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)

    # エッジ（トップ: 赤太線 / その他: グレー細線）
    top_list = [(u, v) for (u, v) in edge_flow_ratio if (u, v) in top_edges]
    other_list = [(u, v) for (u, v) in edge_flow_ratio if (u, v) not in top_edges]
    top_widths = [8 * edge_flow_ratio[(u, v)] / max_ratio for u, v in top_list]
    other_widths = [4 * edge_flow_ratio[(u, v)] / max_ratio for u, v in other_list]

    nx.draw_networkx_edges(G, pos, edgelist=top_list, width=top_widths,
                           edge_color="tomato", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=other_list, width=other_widths,
                           edge_color="gray", alpha=0.5, ax=ax)

    # エッジラベル（%表示）
    edge_labels = {(u, v): f"{r*100:.2f}%" for (u, v), r in edge_flow_ratio.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        ax=ax,
    )

    ax.set_title(
        f"Edge flow ratio (% of total trajectories)  top-{top} in red ({solver})"
    )
    ax.axis("off")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_edge_flow.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="results.json のパス")
    parser.add_argument("--top", type=int, default=3, help="強調するトップエッジ数")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out) if args.out else results_path.parent
    plot_edge_flow(results_path, out_dir, top=args.top)


if __name__ == "__main__":
    main()

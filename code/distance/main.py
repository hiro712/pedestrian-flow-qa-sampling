import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_distances(normalize: bool = True, diag_ratio: float = 0.2):
    """
    Build the distance matrix and scale it by the median or mean.

    Returns:
      distances (np.ndarray): symmetric all-pairs shortest-path weight matrix.
                              Diagonal entries are set to (mean distance)*diag_ratio.
      scale (float): scale factor used for normalization (only when normalize=True)
    """
    # --- 1. Build the graph ---
    G = nx.Graph()
    edges = [
        (8, 9, 20),
        (8, 0, 6),
        (0, 10, 3),
        (8, 6, 30),
        (6, 7, 2),
        (7, 5, 22),
        (5, 4, 9),
        (4, 1, 28),
        (1, 3, 2),
        (3, 2, 15),
    ]
    G.add_weighted_edges_from(edges)

    # --- 2. Shortest-path distance matrix ---
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    nodes = sorted(G.nodes())
    n = len(nodes)
    distances = np.zeros((n, n), dtype=float)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            distances[i, j] = lengths[u][v]

    # --- 3. Diagonal entries ---
    mean_dist = distances.mean()
    np.fill_diagonal(distances, mean_dist * diag_ratio)

    # --- 4. Normalization ---
    scale = 1.0
    if normalize:
        nonzero = distances[distances > 0]
        # Scale by median or mean (median is more robust to outliers)
        scale = np.median(nonzero) if nonzero.size else 1.0
        distances = distances / scale

    return distances, scale


# Example usage
if __name__ == "__main__":
    D, scale = get_distances(normalize=True)
    print("Normalized distance matrix:\n", D)
    print(f"scale (median distance) = {scale:.3f}")

    # Visualize the graph
    G = nx.Graph()
    edges = [
        (8, 9, 20),
        (8, 0, 6),
        (0, 10, 3),
        (8, 6, 30),
        (6, 7, 2),
        (7, 5, 22),
        (5, 4, 9),
        (4, 1, 28),
        (1, 3, 2),
        (3, 2, 15),
    ]
    G.add_weighted_edges_from(edges)

    # Compute the layout (more iterations and tuned k for a more even placement)
    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    # Draw
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6)

    # Show edge labels (weights); white background for readability
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        font_size=12,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    plt.axis("off")
    plt.tight_layout()

    # Save the image
    plt.savefig("graph.png", dpi=300, bbox_inches="tight")
    print("Graph saved as 'graph.png'")

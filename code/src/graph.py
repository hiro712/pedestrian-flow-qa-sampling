import networkx as nx
import numpy as np

# Venue graph (node 0 = outside, 1-10 = internal zones)
EDGES = [
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


def get_distances(normalize: bool = True, diag_ratio: float = 0.2) -> tuple[np.ndarray, float]:
    """
    Return the all-pairs shortest-path distance matrix for the venue graph.

    Returns
    -------
    distances : ndarray, shape (11, 11)
        Inter-node distance matrix (diagonal = mean distance x diag_ratio,
        median-normalized).
    scale : float
        Median value used for normalization (1.0 when normalize=False).
    """
    G = nx.Graph()
    G.add_weighted_edges_from(EDGES)

    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    nodes = sorted(G.nodes())
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            D[i, j] = lengths[u][v]

    mean_dist = D.mean()
    np.fill_diagonal(D, mean_dist * diag_ratio)

    scale = 1.0
    if normalize:
        nonzero = D[D > 0]
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        D = D / scale

    return D, scale

import networkx as nx
import numpy as np

# 会場グラフ（ノード0=外部, 1-10=内部ゾーン）
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
    会場グラフの全対間最短経路距離行列を返す。

    Returns
    -------
    distances : ndarray, shape (11, 11)
        ノード間距離行列（対角=平均距離×diag_ratio、中央値正規化済み）
    scale : float
        正規化に使った中央値（normalize=False のとき 1.0）
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

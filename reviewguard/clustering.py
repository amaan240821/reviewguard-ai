import numpy as np
from .similarity import near_duplicates


def build_clusters(matrix, threshold=0.86):
    n = len(matrix)
    parents = list(range(n))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    edges = near_duplicates(matrix, threshold)
    for a, b, _ in edges:
        parents[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    labels = [-1] * n
    stats = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cid = len(stats)
        for i in members:
            labels[i] = cid
        sub = matrix[np.ix_(members, members)]
        stats.append(
            dict(
                cluster_id=cid,
                size=len(members),
                members=members,
                mean_similarity=float(sub[np.triu_indices(len(members), 1)].mean()),
                edge_count=sum(a in members and b in members for a, b, _ in edges),
            )
        )
    return labels, stats, edges

"""Asset correlation network analysis with graph metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple


def _pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


class CorrelationNetwork:
    """Pairwise correlation graph with graph-theoretic metrics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._assets: List[str] = []
        self._corr: Dict[Tuple[str, str], float] = {}  # (a, b) a < b
        self._edges: Dict[str, Set[str]] = {}

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def build(self, returns: Dict[str, List[float]]) -> None:
        """Compute pairwise Pearson correlations; build edges where |corr| > threshold."""
        self._assets = sorted(returns.keys())
        self._corr = {}
        self._edges = {a: set() for a in self._assets}

        for i, a in enumerate(self._assets):
            for b in self._assets[i + 1:]:
                c = _pearson(returns[a], returns[b])
                self._corr[self._key(a, b)] = c
                if abs(c) > self.threshold:
                    self._edges[a].add(b)
                    self._edges[b].add(a)

    def adjacency_matrix(self) -> List[List[float]]:
        """Correlation adjacency matrix (0 if below threshold)."""
        n = len(self._assets)
        mat = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(self._assets):
            for j, b in enumerate(self._assets):
                if i == j:
                    mat[i][j] = 1.0
                elif i < j:
                    c = self._corr.get(self._key(a, b), 0.0)
                    if abs(c) > self.threshold:
                        mat[i][j] = c
                        mat[j][i] = c
        return mat

    def degree(self, asset: str) -> int:
        """Number of edges connecting asset."""
        return len(self._edges.get(asset, set()))

    def clustering_coefficient(self, asset: str) -> float:
        """Fraction of neighbour pairs that are also connected."""
        neighbors = list(self._edges.get(asset, set()))
        k = len(neighbors)
        if k < 2:
            return 0.0
        connected = 0
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in self._edges.get(neighbors[i], set()):
                    connected += 1
        max_pairs = k * (k - 1) // 2
        return connected / max_pairs if max_pairs > 0 else 0.0

    def minimum_spanning_tree(self) -> List[Tuple[str, str, float]]:
        """Kruskal's MST on |correlation| graph (max-weight = max correlation)."""
        # Sort edges by |correlation| descending (maximum spanning tree on |corr|).
        edges = []
        for (a, b), c in self._corr.items():
            edges.append((abs(c), a, b))
        edges.sort(reverse=True)

        # Union-Find.
        parent = {a: a for a in self._assets}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        mst: List[Tuple[str, str, float]] = []
        for weight, a, b in edges:
            if union(a, b):
                mst.append((a, b, weight))
        return mst

    def communities(self) -> Dict[str, int]:
        """Simple label propagation community detection."""
        labels = {a: i for i, a in enumerate(self._assets)}
        for _ in range(20):
            changed = False
            for a in self._assets:
                neighbors = list(self._edges.get(a, set()))
                if not neighbors:
                    continue
                # Vote for most common label among neighbours.
                counts: Dict[int, int] = {}
                for nb in neighbors:
                    lbl = labels[nb]
                    counts[lbl] = counts.get(lbl, 0) + 1
                best = max(counts, key=lambda k: counts[k])
                if labels[a] != best:
                    labels[a] = best
                    changed = True
            if not changed:
                break
        return labels

    def systemic_risk_score(self) -> float:
        """avg clustering coefficient × avg degree / num_assets."""
        n = len(self._assets)
        if n == 0:
            return 0.0
        avg_cc = sum(self.clustering_coefficient(a) for a in self._assets) / n
        avg_deg = sum(self.degree(a) for a in self._assets) / n
        return avg_cc * avg_deg / n

    def hub_assets(self, top_n: int = 3) -> List[str]:
        """Assets with highest degree."""
        ranked = sorted(self._assets, key=lambda a: self.degree(a), reverse=True)
        return ranked[:top_n]

    def correlation_between(self, a: str, b: str) -> float:
        """Raw Pearson correlation between two assets."""
        return self._corr.get(self._key(a, b), 0.0)

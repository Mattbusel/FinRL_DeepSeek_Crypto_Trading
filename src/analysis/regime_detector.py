"""Market regime detection using HMM-lite (k-means on return/vol features)."""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class MarketRegime:
    name: str
    mean_return: float
    volatility: float
    duration_days: int
    probability: float


class RegimeDetector:
    """Unsupervised market regime detector using k-means clustering."""

    def __init__(self, n_regimes: int = 3, lookback: int = 60):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.centroids: List[Tuple[float, float]] = []  # (mean_ret, vol) centroids
        self._fitted = False

    # ------------------------------------------------------------------
    # Core k-means
    # ------------------------------------------------------------------

    def kmeans_1d(self, data: list, k: int, max_iter: int = 100) -> list:
        """Simple k-means on 1-D data. Returns list of centroid assignments."""
        if not data or k <= 0:
            return []
        n = len(data)
        k = min(k, n)
        # Initialise centroids evenly
        sorted_data = sorted(data)
        step = max(1, n // k)
        centroids = [sorted_data[min(i * step, n - 1)] for i in range(k)]
        labels = [0] * n
        for _ in range(max_iter):
            # Assignment step
            new_labels = []
            for x in data:
                dists = [abs(x - c) for c in centroids]
                new_labels.append(dists.index(min(dists)))
            if new_labels == labels:
                break
            labels = new_labels
            # Update centroids
            for j in range(k):
                members = [data[i] for i in range(n) if labels[i] == j]
                if members:
                    centroids[j] = sum(members) / len(members)
        return labels

    def _rolling_vol(self, returns: list, window: int = 20) -> list:
        """Rolling standard deviation."""
        vols = []
        for i in range(len(returns)):
            start = max(0, i - window + 1)
            window_data = returns[start:i+1]
            if len(window_data) < 2:
                vols.append(0.0)
            else:
                mean = sum(window_data) / len(window_data)
                var = sum((x - mean)**2 for x in window_data) / (len(window_data) - 1)
                vols.append(math.sqrt(var))
        return vols

    def fit(self, returns: list) -> list:
        """Fit k-means on (return, vol) features. Returns regime labels."""
        if len(returns) < self.n_regimes + 1:
            self._fitted = False
            return [0] * len(returns)
        vols = self._rolling_vol(returns)
        # Use returns directly for k-means on the return dimension
        ret_labels = self.kmeans_1d(returns, self.n_regimes)
        # Compute centroids in 2-D feature space
        k = self.n_regimes
        centroid_sum = [[0.0, 0.0] for _ in range(k)]
        centroid_count = [0] * k
        for i, label in enumerate(ret_labels):
            centroid_sum[label][0] += returns[i]
            centroid_sum[label][1] += vols[i]
            centroid_count[label] += 1
        self.centroids = []
        for j in range(k):
            if centroid_count[j] > 0:
                self.centroids.append((
                    centroid_sum[j][0] / centroid_count[j],
                    centroid_sum[j][1] / centroid_count[j],
                ))
            else:
                self.centroids.append((0.0, 0.0))
        self._fitted = True
        return ret_labels

    def predict_current(self, recent_returns: list) -> int:
        """Assign current regime based on nearest centroid."""
        if not self._fitted or not self.centroids or not recent_returns:
            return 0
        mean_ret = sum(recent_returns) / len(recent_returns)
        if len(recent_returns) >= 2:
            mu = mean_ret
            vol = math.sqrt(sum((x - mu)**2 for x in recent_returns) / (len(recent_returns) - 1))
        else:
            vol = 0.0
        dists = [math.sqrt((mean_ret - c[0])**2 + (vol - c[1])**2) for c in self.centroids]
        return dists.index(min(dists))

    def transition_matrix(self, labels: list) -> list:
        """Empirical Markov transition probability matrix."""
        k = self.n_regimes
        counts = [[0.0] * k for _ in range(k)]
        for i in range(len(labels) - 1):
            src = labels[i]
            dst = labels[i + 1]
            if 0 <= src < k and 0 <= dst < k:
                counts[src][dst] += 1.0
        # Normalise rows
        matrix = []
        for row in counts:
            total = sum(row)
            if total > 0:
                matrix.append([x / total for x in row])
            else:
                uniform = 1.0 / k
                matrix.append([uniform] * k)
        return matrix

    def regime_characteristics(self, returns: list, labels: list) -> list:
        """Compute per-regime statistics: mean return, volatility, count."""
        k = self.n_regimes
        buckets: List[List[float]] = [[] for _ in range(k)]
        for ret, label in zip(returns, labels):
            if 0 <= label < k:
                buckets[label].append(ret)
        regimes = []
        for j in range(k):
            data = buckets[j]
            if not data:
                regimes.append(MarketRegime(
                    name=f"regime_{j}", mean_return=0.0, volatility=0.0,
                    duration_days=0, probability=0.0
                ))
                continue
            mean = sum(data) / len(data)
            if len(data) >= 2:
                var = sum((x - mean)**2 for x in data) / (len(data) - 1)
                vol = math.sqrt(var)
            else:
                vol = 0.0
            prob = len(data) / len(returns) if returns else 0.0
            regimes.append(MarketRegime(
                name=f"regime_{j}",
                mean_return=mean,
                volatility=vol,
                duration_days=len(data),
                probability=prob,
            ))
        return regimes

    def regime_persistence(self, labels: list) -> dict:
        """Average run length per regime."""
        k = self.n_regimes
        durations: Dict[int, List[int]] = {j: [] for j in range(k)}
        if not labels:
            return {j: 0.0 for j in range(k)}
        current = labels[0]
        run = 1
        for label in labels[1:]:
            if label == current:
                run += 1
            else:
                durations[current].append(run)
                current = label
                run = 1
        durations[current].append(run)
        result = {}
        for j in range(k):
            runs = durations[j]
            result[j] = sum(runs) / len(runs) if runs else 0.0
        return result

    def current_regime_probability(self, recent_returns: list) -> list:
        """Soft assignment: inverse-distance weights to each centroid."""
        if not self._fitted or not self.centroids or not recent_returns:
            k = self.n_regimes
            return [1.0 / k] * k
        mean_ret = sum(recent_returns) / len(recent_returns)
        if len(recent_returns) >= 2:
            mu = mean_ret
            vol = math.sqrt(sum((x - mu)**2 for x in recent_returns) / (len(recent_returns) - 1))
        else:
            vol = 0.0
        dists = [math.sqrt((mean_ret - c[0])**2 + (vol - c[1])**2) + 1e-9 for c in self.centroids]
        inv_dists = [1.0 / d for d in dists]
        total = sum(inv_dists)
        return [w / total for w in inv_dists]

    def bull_bear_sideways(self, returns: list) -> list:
        """3-regime classification: 0=bear, 1=sideways, 2=bull."""
        if not returns:
            return []
        n_orig = self.n_regimes
        self.n_regimes = 3
        labels = self.fit(returns)
        self.n_regimes = n_orig
        if not self.centroids:
            return labels
        # Sort centroids by mean return: lowest=bear(0), middle=sideways(1), highest=bull(2)
        sorted_idx = sorted(range(len(self.centroids)), key=lambda i: self.centroids[i][0])
        remap = {old: new for new, old in enumerate(sorted_idx)}
        relabeled = [remap.get(lbl, lbl) for lbl in labels]
        # Restore original n_regimes
        self.n_regimes = n_orig
        return relabeled

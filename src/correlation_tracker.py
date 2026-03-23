"""Rolling Pearson correlation matrix for crypto assets.

This module provides a lightweight, dependency-optional correlation tracker
that maintains a sliding window of returns per asset and computes pairwise
Pearson correlations on demand.  It is designed for crypto markets where
assets can be added dynamically and data arrives as streaming updates.

Usage::

    from src.correlation_tracker import CorrelationTracker

    tracker = CorrelationTracker(window=30)
    tracker.update("BTC", 0.012, timestamp=1.0)
    tracker.update("ETH", 0.009, timestamp=1.0)
    tracker.update("BTC", -0.003, timestamp=2.0)
    tracker.update("ETH", -0.001, timestamp=2.0)

    corr = tracker.correlation("BTC", "ETH")
    matrix = tracker.full_matrix()
    pairs = tracker.highly_correlated(threshold=0.8)
    score = tracker.diversification_score()
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional pandas import — fall back to dict-of-dicts if unavailable
# ---------------------------------------------------------------------------
try:
    import pandas as _pd  # noqa: F401

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation for equal-length lists.

    Returns ``None`` if there are fewer than 2 observations or if either
    series has zero variance (constant series).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return None

    return cov / denom


class CorrelationTracker:
    """Rolling Pearson correlation matrix for a dynamic set of assets.

    Parameters
    ----------
    window:
        Number of most-recent return observations to retain per asset.
        Older observations are evicted automatically (FIFO).
    assets:
        Optional pre-declared list of asset names.  Additional assets
        can always be added dynamically via :meth:`update`.

    Notes
    -----
    * Correlations are computed lazily (on demand) from the rolling window.
    * The tracker stores only raw returns, not covariance matrices, so
      memory scales as ``O(window * n_assets)``.
    * All public methods are **not** thread-safe.
    """

    def __init__(
        self,
        window: int = 30,
        assets: Optional[List[str]] = None,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._window = window
        # asset_name -> deque of (timestamp, return) pairs, capped at window
        self._returns: Dict[str, deque] = {}
        if assets:
            for a in assets:
                self._ensure_asset(a)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_asset(self, asset: str) -> None:
        if asset not in self._returns:
            self._returns[asset] = deque(maxlen=self._window)

    def _return_series(self, asset: str) -> List[float]:
        """Return the plain list of return values for *asset*."""
        return [r for _, r in self._returns.get(asset, [])]

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(self, asset: str, return_: float, timestamp: float) -> None:
        """Append a new return observation for *asset*.

        Parameters
        ----------
        asset:
            Identifier for the crypto asset (e.g. ``"BTC"``, ``"ETH"``).
        return_:
            Fractional return for the period (e.g. ``0.012`` for +1.2 %).
        timestamp:
            Unix timestamp or any monotonically increasing float used for
            reference (currently stored but not used for gap detection).
        """
        self._ensure_asset(asset)
        self._returns[asset].append((timestamp, return_))

    # ------------------------------------------------------------------
    # Correlation queries
    # ------------------------------------------------------------------

    def correlation(self, asset_a: str, asset_b: str) -> Optional[float]:
        """Pearson correlation between two assets over the rolling window.

        Returns ``None`` if either asset has fewer than 2 observations or
        if their return series are constant (zero variance).
        """
        if asset_a not in self._returns or asset_b not in self._returns:
            return None

        series_a = self._return_series(asset_a)
        series_b = self._return_series(asset_b)

        # Align to the shorter series length (both windows are the same max size)
        min_len = min(len(series_a), len(series_b))
        if min_len < 2:
            return None

        return _pearson(series_a[-min_len:], series_b[-min_len:])

    def full_matrix(self):
        """Compute the full pairwise correlation matrix.

        Returns
        -------
        pandas.DataFrame
            When pandas is available — symmetric N×N DataFrame with asset
            names as both row and column labels, diagonal entries = 1.0.
        dict[str, dict[str, Optional[float]]]
            When pandas is unavailable — nested dict with the same data.
        """
        assets = sorted(self._returns.keys())
        n = len(assets)

        matrix: Dict[str, Dict[str, Optional[float]]] = {
            a: {} for a in assets
        }

        for i in range(n):
            a = assets[i]
            matrix[a][a] = 1.0
            for j in range(i + 1, n):
                b = assets[j]
                corr = self.correlation(a, b)
                matrix[a][b] = corr
                matrix[b][a] = corr

        if _PANDAS_AVAILABLE:
            import pandas as pd

            return pd.DataFrame(matrix, index=assets, columns=assets)

        return matrix

    def highly_correlated(
        self, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Return asset pairs whose correlation exceeds *threshold*.

        Parameters
        ----------
        threshold:
            Minimum absolute Pearson correlation to qualify (default 0.8).

        Returns
        -------
        list[tuple[str, str, float]]
            Sorted list of ``(asset_a, asset_b, correlation)`` tuples
            where ``|correlation| >= threshold``, ordered by descending
            absolute correlation.
        """
        assets = sorted(self._returns.keys())
        result: List[Tuple[str, str, float]] = []

        for i, a in enumerate(assets):
            for b in assets[i + 1 :]:
                corr = self.correlation(a, b)
                if corr is not None and abs(corr) >= threshold:
                    result.append((a, b, corr))

        result.sort(key=lambda t: abs(t[2]), reverse=True)
        return result

    def diversification_score(self) -> float:
        """Portfolio diversification score in ``[0, 1]``.

        Defined as ``1 - mean(|corr(a, b)|)`` over all unique asset pairs.
        A score of 1.0 means all assets are uncorrelated (maximum
        diversification); 0.0 means perfect positive correlation everywhere.

        Returns ``1.0`` if there is only one asset (trivially diversified).
        """
        assets = sorted(self._returns.keys())
        n = len(assets)

        if n < 2:
            return 1.0

        total = 0.0
        count = 0

        for i, a in enumerate(assets):
            for b in assets[i + 1 :]:
                corr = self.correlation(a, b)
                if corr is not None:
                    total += abs(corr)
                    count += 1

        if count == 0:
            return 1.0

        return 1.0 - (total / count)

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def assets(self) -> List[str]:
        """Sorted list of all tracked asset names."""
        return sorted(self._returns.keys())

    @property
    def window(self) -> int:
        """Rolling window size."""
        return self._window

    def observation_count(self, asset: str) -> int:
        """Number of observations stored for *asset* (up to ``window``)."""
        return len(self._returns.get(asset, []))

    def reset(self, asset: Optional[str] = None) -> None:
        """Clear history for *asset*, or all assets if ``None``."""
        if asset is None:
            self._returns.clear()
        elif asset in self._returns:
            self._returns[asset].clear()

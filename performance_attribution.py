"""Strategy performance attribution.

Implements Brinson-Hood-Beebower attribution, Jensen's alpha, rolling alpha,
information ratio, and related performance decomposition tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Attribution Period ────────────────────────────────────────────────────────

@dataclass
class AttributionPeriod:
    """Performance attribution results for a single period."""
    start_date: str
    end_date: str
    total_return: float
    benchmark_return: float
    alpha: float
    selection_return: float
    timing_return: float


# ── OLS Helper ────────────────────────────────────────────────────────────────

def _ols_slope_intercept(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Simple OLS: returns (slope, intercept) for y = slope * x + intercept."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (n - ddof)
    return math.sqrt(variance)


# ── Performance Attributor ────────────────────────────────────────────────────

class PerformanceAttributor:
    """Tools for decomposing portfolio performance relative to a benchmark.

    All methods are stateless and accept plain Python lists or dicts.

    Example::

        attr = PerformanceAttributor()
        alpha = attr.compute_alpha(portfolio_returns, benchmark_returns)
        ir = attr.information_ratio(portfolio_returns, benchmark_returns)
    """

    def compute_alpha(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        """Jensen's alpha via OLS regression.

        Fits: R_p = alpha + beta * R_b + epsilon
        Returns the intercept (alpha).

        Args:
            portfolio_returns: Periodic portfolio returns.
            benchmark_returns: Corresponding benchmark returns.

        Returns:
            Jensen's alpha (annualized equivalent of the OLS intercept).
        """
        n = min(len(portfolio_returns), len(benchmark_returns))
        if n < 2:
            return 0.0
        beta, alpha = _ols_slope_intercept(
            benchmark_returns[:n], portfolio_returns[:n]
        )
        return alpha

    def selection_effect(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        asset_returns: Dict[str, float],
    ) -> float:
        """Brinson selection effect: Σ (w_p - w_b) * r_asset.

        Measures the value added by selecting assets differently from the benchmark,
        using benchmark weights as the baseline.

        Args:
            portfolio_weights: {asset: weight} for the portfolio.
            benchmark_weights: {asset: weight} for the benchmark.
            asset_returns: {asset: return} for each asset.

        Returns:
            Selection return (scalar).
        """
        assets = set(portfolio_weights) | set(benchmark_weights)
        effect = 0.0
        for asset in assets:
            wp = portfolio_weights.get(asset, 0.0)
            wb = benchmark_weights.get(asset, 0.0)
            r = asset_returns.get(asset, 0.0)
            effect += (wp - wb) * r
        return effect

    def timing_effect(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        asset_returns: Dict[str, float],
        benchmark_return: float,
    ) -> float:
        """Brinson timing (allocation) effect: Σ (w_p - w_b) * (r_asset - r_benchmark).

        Measures the value added by over- or under-weighting assets relative to
        how those assets performed versus the benchmark.

        Args:
            portfolio_weights: {asset: weight} for the portfolio.
            benchmark_weights: {asset: weight} for the benchmark.
            asset_returns: {asset: return} for each asset.
            benchmark_return: Overall benchmark return.

        Returns:
            Timing/allocation return (scalar).
        """
        assets = set(portfolio_weights) | set(benchmark_weights)
        effect = 0.0
        for asset in assets:
            wp = portfolio_weights.get(asset, 0.0)
            wb = benchmark_weights.get(asset, 0.0)
            r = asset_returns.get(asset, 0.0)
            effect += (wp - wb) * (r - benchmark_return)
        return effect

    def brinson_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        asset_returns: Dict[str, float],
        benchmark_return: float,
    ) -> Dict[str, float]:
        """Full Brinson-Hood-Beebower attribution decomposition.

        Decomposes active return into:
        - allocation: over/under-weighting assets (timing effect)
        - selection: picking assets with different returns within each bucket
        - interaction: cross-term between allocation and selection

        Args:
            portfolio_weights: {asset: weight} for the portfolio.
            benchmark_weights: {asset: weight} for the benchmark.
            asset_returns: {asset: return} for each asset.
            benchmark_return: Overall benchmark return.

        Returns:
            Dict with keys 'allocation', 'selection', 'interaction', 'active_return'.
        """
        assets = set(portfolio_weights) | set(benchmark_weights)
        allocation = 0.0
        selection = 0.0
        interaction = 0.0

        for asset in assets:
            wp = portfolio_weights.get(asset, 0.0)
            wb = benchmark_weights.get(asset, 0.0)
            r_asset = asset_returns.get(asset, 0.0)

            allocation += (wp - wb) * (benchmark_return)
            selection += wb * (r_asset - benchmark_return)
            interaction += (wp - wb) * (r_asset - benchmark_return)

        # Active return = portfolio return - benchmark return
        portfolio_return = sum(
            portfolio_weights.get(a, 0.0) * asset_returns.get(a, 0.0)
            for a in assets
        )
        active_return = portfolio_return - benchmark_return

        return {
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "active_return": active_return,
        }

    def rolling_alpha(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
        window: int = 30,
    ) -> List[float]:
        """Compute rolling Jensen's alpha over a sliding window.

        Args:
            portfolio_returns: Full series of portfolio returns.
            benchmark_returns: Full series of benchmark returns.
            window: Rolling window size in periods.

        Returns:
            List of rolling alpha values. The first (window-1) entries are NaN
            represented as 0.0 (insufficient data).
        """
        n = min(len(portfolio_returns), len(benchmark_returns))
        alphas = []
        for i in range(n):
            if i < window - 1:
                alphas.append(0.0)
            else:
                p_window = portfolio_returns[i - window + 1: i + 1]
                b_window = benchmark_returns[i - window + 1: i + 1]
                alphas.append(self.compute_alpha(p_window, b_window))
        return alphas

    def information_ratio(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        """Information ratio = alpha / tracking_error.

        Where tracking_error is the standard deviation of active returns
        (portfolio_return - benchmark_return).

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series.

        Returns:
            Information ratio. Returns 0.0 if tracking error is zero.
        """
        n = min(len(portfolio_returns), len(benchmark_returns))
        if n < 2:
            return 0.0

        active_returns = [
            portfolio_returns[i] - benchmark_returns[i]
            for i in range(n)
        ]
        alpha = _mean(active_returns)
        tracking_error = _std(active_returns, ddof=1)

        if tracking_error == 0.0:
            return 0.0

        return alpha / tracking_error

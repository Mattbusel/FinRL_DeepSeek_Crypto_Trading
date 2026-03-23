"""
Sharpe ratio portfolio optimisation.

Provides gradient ascent, random search, efficient frontier construction,
max-Sharpe, minimum-variance, and risk-parity weight solvers using only
the Python standard library.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _portfolio_return(returns: List[List[float]], weights: List[float]) -> float:
    """Mean portfolio return: Σ_i w_i * mean(r_i)."""
    n_assets = len(returns)
    means = [sum(r) / max(len(r), 1) for r in returns]
    return _dot(weights, means)


def _portfolio_variance(returns: List[List[float]], weights: List[float]) -> float:
    """Portfolio variance using the sample covariance matrix."""
    n_assets = len(returns)
    n_obs = min(len(r) for r in returns) if returns else 0
    if n_obs < 2:
        return 0.0

    means = [sum(r[:n_obs]) / n_obs for r in returns]
    var = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            cov = sum(
                (returns[i][t] - means[i]) * (returns[j][t] - means[j])
                for t in range(n_obs)
            ) / (n_obs - 1)
            var += weights[i] * weights[j] * cov
    return max(var, 0.0)


def _normalise_weights(weights: List[float]) -> List[float]:
    """Project weights to the unit simplex (all non-negative, sum to 1)."""
    clipped = [max(w, 0.0) for w in weights]
    total = sum(clipped)
    if total < 1e-12:
        n = len(weights)
        return [1.0 / n] * n
    return [w / total for w in clipped]


def sharpe_ratio(
    returns: List[List[float]],
    weights: List[float],
    rf: float = 0.04,
) -> float:
    """Annualised Sharpe ratio.

    Assumes daily returns (252 trading days/year).

    Args:
        returns: List of per-asset daily return series.
        weights: Portfolio weights (must sum to 1).
        rf:      Annual risk-free rate.

    Returns:
        Sharpe ratio. Returns 0.0 if volatility is negligible.
    """
    port_ret = _portfolio_return(returns, weights) * 252
    port_var = _portfolio_variance(returns, weights) * 252
    port_vol = math.sqrt(port_var)
    if port_vol < 1e-12:
        return 0.0
    return (port_ret - rf) / port_vol


def portfolio_stats(
    returns: List[List[float]],
    weights: List[float],
) -> Dict[str, float]:
    """Compute key portfolio statistics.

    Args:
        returns: Per-asset daily return series.
        weights: Portfolio weights.

    Returns:
        Dict with keys: 'return', 'vol', 'sharpe', 'max_dd'.
    """
    w = _normalise_weights(weights)
    n_obs = min(len(r) for r in returns) if returns else 0

    port_ret_ann = _portfolio_return(returns, w) * 252
    port_vol_ann = math.sqrt(_portfolio_variance(returns, w) * 252)
    sr = sharpe_ratio(returns, w)

    # Max drawdown on daily portfolio return series
    max_dd = 0.0
    if n_obs > 1:
        port_rets = [
            sum(returns[a][t] * w[a] for a in range(len(returns)))
            for t in range(n_obs)
        ]
        cumulative = 1.0
        peak = 1.0
        for r in port_rets:
            cumulative *= (1.0 + r)
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak
            if dd > max_dd:
                max_dd = dd

    return {
        "return": port_ret_ann,
        "vol": port_vol_ann,
        "sharpe": sr,
        "max_dd": max_dd,
    }


# ─── SHARPE OPTIMISER ────────────────────────────────────────────────────────

class SharpeOptimizer:
    """Maximises the Sharpe ratio of a multi-asset portfolio.

    Args:
        returns:   List of per-asset daily return series (list of lists).
        rf_rate:   Annual risk-free rate.
        max_iter:  Maximum iterations for gradient ascent.
    """

    def __init__(
        self,
        returns: List[List[float]],
        rf_rate: float = 0.04,
        max_iter: int = 1_000,
    ) -> None:
        self.returns = returns
        self.rf_rate = rf_rate
        self.max_iter = max_iter
        self.n_assets = len(returns)

    def _uniform_weights(self) -> List[float]:
        return [1.0 / self.n_assets] * self.n_assets

    def gradient_ascent(self, lr: float = 0.01) -> List[float]:
        """Gradient ascent on the Sharpe ratio.

        Uses finite-difference gradient estimation.

        Args:
            lr: Step size for weight updates.

        Returns:
            Optimised weights summing to 1.
        """
        weights = self._uniform_weights()
        eps = 1e-5
        best_sharpe = sharpe_ratio(self.returns, weights, self.rf_rate)
        best_weights = list(weights)

        for _ in range(self.max_iter):
            grad = []
            for i in range(self.n_assets):
                w_plus = list(weights)
                w_plus[i] += eps
                w_plus = _normalise_weights(w_plus)
                sr_plus = sharpe_ratio(self.returns, w_plus, self.rf_rate)
                grad.append((sr_plus - best_sharpe) / eps)

            # Ascent step
            new_weights = [weights[i] + lr * grad[i] for i in range(self.n_assets)]
            new_weights = _normalise_weights(new_weights)
            new_sharpe = sharpe_ratio(self.returns, new_weights, self.rf_rate)

            if new_sharpe > best_sharpe:
                best_sharpe = new_sharpe
                best_weights = list(new_weights)

            weights = new_weights

        return best_weights

    def random_search(
        self,
        n_portfolios: int = 1_000,
        seed: int = 42,
    ) -> List[float]:
        """Find the best weights by random sampling on the simplex.

        Args:
            n_portfolios: Number of random portfolios to evaluate.
            seed:         RNG seed.

        Returns:
            Weights with the highest Sharpe ratio found.
        """
        rng = random.Random(seed)
        best_sharpe = -math.inf
        best_weights = self._uniform_weights()

        for _ in range(n_portfolios):
            # Sample from Dirichlet(1,…,1) via Gamma(1) trick
            gammas = [-math.log(rng.random() + 1e-12) for _ in range(self.n_assets)]
            total = sum(gammas)
            w = [g / total for g in gammas]
            sr = sharpe_ratio(self.returns, w, self.rf_rate)
            if sr > best_sharpe:
                best_sharpe = sr
                best_weights = w

        return best_weights

    def efficient_frontier(self, n_points: int = 20) -> List[Tuple[float, float, float, List[float]]]:
        """Trace the mean-variance efficient frontier.

        Parameterises portfolios by targeting n_points return levels from
        the minimum to the maximum asset mean return.

        Args:
            n_points: Number of points along the frontier.

        Returns:
            List of (annualised_return, annualised_vol, sharpe, weights) tuples.
        """
        means = [sum(r) / max(len(r), 1) * 252 for r in self.returns]
        min_ret = min(means)
        max_ret = max(means)

        if abs(max_ret - min_ret) < 1e-10:
            w = self._uniform_weights()
            stats = portfolio_stats(self.returns, w)
            return [(stats["return"], stats["vol"], stats["sharpe"], w)] * n_points

        frontier = []
        for i in range(n_points):
            target = min_ret + (max_ret - min_ret) * i / max(n_points - 1, 1)
            # Minimum-variance portfolio for this return target (unconstrained)
            w = self._min_var_for_target(target / 252)
            stats = portfolio_stats(self.returns, w)
            frontier.append((stats["return"], stats["vol"], stats["sharpe"], w))

        return frontier

    def _min_var_for_target(self, target_daily_ret: float) -> List[float]:
        """Approximate minimum-variance portfolio near a target return."""
        rng = random.Random(0)
        best_var = math.inf
        best_w = self._uniform_weights()

        for _ in range(200):
            gammas = [-math.log(rng.random() + 1e-12) for _ in range(self.n_assets)]
            total = sum(gammas)
            w = [g / total for g in gammas]
            ret = _portfolio_return(self.returns, w)
            # Penalise deviation from target return
            penalty = 1e6 * (ret - target_daily_ret) ** 2
            var = _portfolio_variance(self.returns, w) + penalty
            if var < best_var:
                best_var = var
                best_w = w

        return best_w

    def max_sharpe_weights(self) -> List[float]:
        """Return weights that maximise the Sharpe ratio.

        Uses a combination of gradient ascent and random search.

        Returns:
            Optimised weight vector.
        """
        w_grad = self.gradient_ascent()
        w_rand = self.random_search()
        if sharpe_ratio(self.returns, w_grad, self.rf_rate) >= \
           sharpe_ratio(self.returns, w_rand, self.rf_rate):
            return w_grad
        return w_rand

    def min_variance_weights(self) -> List[float]:
        """Return the minimum-variance portfolio weights.

        Uses random search on the simplex minimising portfolio variance.

        Returns:
            Minimum-variance weight vector.
        """
        rng = random.Random(99)
        best_var = math.inf
        best_w = self._uniform_weights()

        for _ in range(max(self.max_iter, 2_000)):
            gammas = [-math.log(rng.random() + 1e-12) for _ in range(self.n_assets)]
            total = sum(gammas)
            w = [g / total for g in gammas]
            var = _portfolio_variance(self.returns, w)
            if var < best_var:
                best_var = var
                best_w = w

        return best_w

    def risk_parity_weights(self) -> List[float]:
        """Return risk-parity (equal risk contribution) weights.

        Approximates equal risk contributions by setting w_i ∝ 1/σ_i
        where σ_i is the standard deviation of asset i.

        Returns:
            Risk-parity weight vector.
        """
        vols = []
        for asset_returns in self.returns:
            if len(asset_returns) < 2:
                vols.append(1.0)
                continue
            mean = sum(asset_returns) / len(asset_returns)
            var = sum((r - mean) ** 2 for r in asset_returns) / (len(asset_returns) - 1)
            vols.append(math.sqrt(max(var, 1e-12)))

        inv_vol = [1.0 / v for v in vols]
        total = sum(inv_vol)
        return [iv / total for iv in inv_vol]


__all__ = ["sharpe_ratio", "portfolio_stats", "SharpeOptimizer"]

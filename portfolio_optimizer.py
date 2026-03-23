"""Multi-asset crypto portfolio optimization.

Implements classical portfolio optimization algorithms in pure NumPy, with a
backtesting harness for evaluating historical performance.

Algorithms:
- Mean-variance optimization (projected gradient ascent)
- Equal risk contribution / risk parity
- Maximum Sharpe ratio (gradient descent on negative Sharpe)
- Minimum variance

All weight vectors satisfy sum(w) = 1, w >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optimization algorithms
# ---------------------------------------------------------------------------


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w: sum=1, w>=0}.

    Uses the O(n log n) algorithm from Duchi et al. (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def mean_variance_optimize(
    returns_matrix: np.ndarray,
    risk_aversion: float = 2.0,
    n_iter: int = 2000,
    lr: float = 1e-2,
) -> np.ndarray:
    """Mean-variance portfolio weights via projected gradient ascent.

    Maximises the objective:
        U(w) = μ'w - (λ/2) · w'Σw
    subject to sum(w) = 1, w >= 0.

    Args:
        returns_matrix: (T, N) array of asset returns.
        risk_aversion: Lambda (λ) controlling the risk penalty.
        n_iter: Number of gradient steps.
        lr: Learning rate.

    Returns:
        (N,) weight vector summing to 1 with non-negative elements.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.array([[returns_matrix.var()]])
    n = len(mu)
    w = np.ones(n) / n

    for _ in range(n_iter):
        grad = mu - risk_aversion * cov @ w
        w = w + lr * grad
        w = _project_simplex(w)

    return w


def equal_risk_contribution(
    cov_matrix: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Risk parity weights: each asset contributes equally to portfolio variance.

    Uses iterative proportional rescaling (Maillard et al. 2010 approach).

    Args:
        cov_matrix: (N, N) covariance matrix.
        n_iter: Maximum iterations.
        tol: Convergence tolerance on weight change.

    Returns:
        (N,) weight vector.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]
    w = np.ones(n) / n
    target_risk = 1.0 / n  # equal risk contribution

    for _ in range(n_iter):
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-12:
            break
        marginal_risk = cov @ w / sigma_p
        risk_contrib = w * marginal_risk
        # Scale each weight toward target risk contribution
        w_new = w * target_risk / (risk_contrib + 1e-12)
        w_new = np.maximum(w_new, 0.0)
        total = w_new.sum()
        if total < 1e-12:
            break
        w_new /= total
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def max_sharpe(
    returns_matrix: np.ndarray,
    risk_free: float = 0.0,
    n_iter: int = 3000,
    lr: float = 5e-3,
) -> np.ndarray:
    """Maximum Sharpe ratio weights via gradient ascent on Sharpe.

    Args:
        returns_matrix: (T, N) array of asset returns.
        risk_free: Risk-free rate per period.
        n_iter: Number of gradient steps.
        lr: Learning rate.

    Returns:
        (N,) weight vector.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.array([[returns_matrix.var()]])
    n = len(mu)
    w = np.ones(n) / n

    for _ in range(n_iter):
        port_ret = w @ mu - risk_free
        port_var = w @ cov @ w
        port_std = np.sqrt(max(port_var, 1e-12))
        # d(Sharpe)/dw
        grad_ret = mu
        grad_std = (cov @ w) / port_std
        sharpe = port_ret / port_std
        grad = (grad_ret * port_std - port_ret * grad_std) / (port_std ** 2)
        w = w + lr * grad
        w = _project_simplex(w)

    return w


def min_variance(
    cov_matrix: np.ndarray,
    n_iter: int = 2000,
    lr: float = 1e-2,
) -> np.ndarray:
    """Minimum variance portfolio weights.

    Args:
        cov_matrix: (N, N) covariance matrix.
        n_iter: Number of projected gradient steps.
        lr: Learning rate.

    Returns:
        (N,) weight vector.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(n_iter):
        grad = -2.0 * cov @ w  # gradient of -variance (ascent = minimize variance)
        w = w + lr * grad
        w = _project_simplex(w)

    return w


# ---------------------------------------------------------------------------
# Portfolio Backtester
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Full set of backtesting performance statistics."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    turnover: float


class PortfolioBacktester:
    """Evaluates a portfolio weighting scheme on historical returns data.

    Args:
        risk_free_rate: Annual risk-free rate (used for Sharpe / Sortino).
        periods_per_year: Number of return periods per year (e.g. 252 for daily).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def backtest(
        self,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> BacktestResult:
        """Run a static-weight backtest.

        Args:
            weights: (N,) portfolio weights (must sum to ~1).
            returns_matrix: (T, N) asset returns matrix.

        Returns:
            ``BacktestResult`` with performance statistics.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=float)
        weights = np.asarray(weights, dtype=float)

        port_returns = returns_matrix @ weights  # (T,)

        # Total return
        total_return = float(np.prod(1.0 + port_returns) - 1.0)

        # Sharpe ratio (annualised)
        mean_r = port_returns.mean()
        std_r = port_returns.std(ddof=1) if len(port_returns) > 1 else 1e-12
        rf_per_period = self.risk_free_rate / self.periods_per_year
        sharpe = (mean_r - rf_per_period) / max(std_r, 1e-12) * np.sqrt(self.periods_per_year)

        # Max drawdown
        equity = np.cumprod(1.0 + port_returns)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-12)
        max_dd = float(dd.max())

        # Calmar ratio
        ann_return = (1.0 + total_return) ** (self.periods_per_year / max(len(port_returns), 1)) - 1.0
        calmar = ann_return / max(max_dd, 1e-12)

        # Sortino ratio
        downside = port_returns[port_returns < rf_per_period] - rf_per_period
        downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-12
        sortino = (mean_r - rf_per_period) / max(downside_std, 1e-12) * np.sqrt(self.periods_per_year)

        # Win rate
        win_rate = float((port_returns > 0).mean())

        # Turnover (N/A for static weights — 0 by definition)
        turnover = 0.0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=max_dd,
            calmar_ratio=float(calmar),
            sortino_ratio=float(sortino),
            win_rate=win_rate,
            turnover=turnover,
        )

    def backtest_weights_history(
        self,
        weights_history: pd.DataFrame,
        returns_matrix: pd.DataFrame,
    ) -> BacktestResult:
        """Backtest with time-varying weights.

        Args:
            weights_history: (T, N) DataFrame of portfolio weights per period.
            returns_matrix: (T, N) DataFrame of asset returns per period.

        Returns:
            ``BacktestResult`` reflecting the dynamic strategy.
        """
        aligned_w = weights_history.reindex(returns_matrix.index).ffill().fillna(0.0)
        port_returns = (aligned_w * returns_matrix).sum(axis=1).values

        # Turnover: average L1 weight change
        weight_changes = aligned_w.diff().abs().sum(axis=1)
        turnover = float(weight_changes.mean())

        total_return = float(np.prod(1.0 + port_returns) - 1.0)

        mean_r = port_returns.mean()
        std_r = port_returns.std(ddof=1) if len(port_returns) > 1 else 1e-12
        rf_per_period = self.risk_free_rate / self.periods_per_year
        sharpe = (mean_r - rf_per_period) / max(std_r, 1e-12) * np.sqrt(self.periods_per_year)

        equity = np.cumprod(1.0 + port_returns)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-12)
        max_dd = float(dd.max())

        ann_return = (1.0 + total_return) ** (self.periods_per_year / max(len(port_returns), 1)) - 1.0
        calmar = ann_return / max(max_dd, 1e-12)

        downside = port_returns[port_returns < rf_per_period] - rf_per_period
        downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-12
        sortino = (mean_r - rf_per_period) / max(downside_std, 1e-12) * np.sqrt(self.periods_per_year)

        win_rate = float((port_returns > 0).mean())

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=max_dd,
            calmar_ratio=float(calmar),
            sortino_ratio=float(sortino),
            win_rate=win_rate,
            turnover=turnover,
        )


# ---------------------------------------------------------------------------
# Rebalancing schedule
# ---------------------------------------------------------------------------


def rebalance_schedule(
    weights_history: pd.DataFrame,
    threshold: float,
) -> list[int]:
    """Identify rebalancing dates when any weight drifts beyond threshold.

    Args:
        weights_history: (T, N) DataFrame of portfolio weights indexed by date.
        threshold: Maximum allowed absolute deviation from the initial target weight.

    Returns:
        List of integer row indices (within ``weights_history``) where
        rebalancing should occur.
    """
    if weights_history.empty:
        return []

    rebalance_dates: list[int] = []
    target = weights_history.iloc[0].values.copy()

    for i in range(1, len(weights_history)):
        current = weights_history.iloc[i].values
        drift = np.abs(current - target).max()
        if drift > threshold:
            rebalance_dates.append(i)
            target = current.copy()  # reset target to current after rebalance

    return rebalance_dates

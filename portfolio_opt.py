"""Portfolio optimisation methods for multi-asset crypto allocation.

Implements four optimisation strategies plus a periodic rebalancer:

1. ``MeanVarianceOptimizer``   -- Markowitz mean-variance with optional
                                  Black-Litterman view injection.
2. ``HierarchicalRiskParity``  -- Clusters assets by correlation, allocates
                                  equal risk across clusters (López de Prado).
3. ``MinimumCorrelation``      -- Minimises the average pairwise correlation
                                  of the portfolio.
4. ``BlackLittermanViews``     -- Converts RL agent Q-value signals into
                                  Bayesian prior views on expected returns.
5. ``PortfolioRebalancer``     -- Periodic rebalancing with transaction-cost
                                  awareness and configurable strategy selection.

All optimisers share the same interface:

    weights = optimizer.allocate(returns)   # np.ndarray shape (T, N)

where the returned ``weights`` is a ``numpy.ndarray`` of shape ``(N,)`` that
sums to 1.0.

Example::

    from portfolio_opt import (
        HierarchicalRiskParity,
        MeanVarianceOptimizer,
        BlackLittermanViews,
        PortfolioRebalancer,
    )
    import numpy as np

    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.02, (120, 3)).astype(np.float32)

    hrp = HierarchicalRiskParity(assets=["BTC", "ETH", "SOL"])
    weights = hrp.allocate(returns)
    print(weights)  # e.g. [0.45, 0.33, 0.22]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from exceptions import DataFetchError, ModelError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_LOOKBACK: int = 10
_EPSILON: float = 1e-8


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _validate_returns(returns: npt.NDArray[np.float32], n_assets: int) -> None:
    """Raise if *returns* is too small or has wrong number of assets.

    Args:
        returns: Return matrix of shape ``(T, N)``.
        n_assets: Expected number of asset columns.

    Raises:
        DataFetchError: If the matrix is malformed.
    """
    if returns.ndim != 2:
        raise DataFetchError(
            f"returns must be 2-D (T × N), got shape {returns.shape}"
        )
    T, N = returns.shape
    if N != n_assets:
        raise DataFetchError(
            f"returns has {N} asset columns; expected {n_assets}."
        )
    if T < _MIN_LOOKBACK:
        raise DataFetchError(
            f"Need at least {_MIN_LOOKBACK} rows; got {T}."
        )


def _cov_shrinkage(returns: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Ledoit-Wolf-style linear shrinkage of the sample covariance matrix.

    Shrinks towards the diagonal of equal variances to improve
    out-of-sample stability.

    Args:
        returns: Return matrix of shape ``(T, N)``.

    Returns:
        Shrunk covariance matrix of shape ``(N, N)``.
    """
    T, N = returns.shape
    S = np.cov(returns.T, ddof=1)
    # Shrinkage intensity: Oracle approximating shrinkage (OAS simple form)
    mu_target = np.trace(S) / N
    F = mu_target * np.eye(N)
    rho = min(1.0, (1.0 - 2.0 / N) / ((T + 1.0 - 2.0 / N) + _EPSILON))
    return (1.0 - rho) * S + rho * F


def _normalise(weights: npt.NDArray) -> npt.NDArray[np.float64]:
    """Normalise *weights* to sum to 1.0; return equal weights on failure."""
    total = float(np.sum(weights))
    if total < _EPSILON or not np.isfinite(total):
        n = len(weights)
        return np.full(n, 1.0 / n)
    return np.asarray(weights, dtype=np.float64) / total


# ---------------------------------------------------------------------------
# Black-Litterman Views
# ---------------------------------------------------------------------------


@dataclass
class BLViewSet:
    """Encapsulates a set of Black-Litterman views.

    Attributes:
        P: Pick matrix of shape ``(K, N)`` — each row selects the assets
           covered by one view.
        Q: View returns column vector of shape ``(K,)``.
        omega: View uncertainty diagonal of shape ``(K,)``; larger = less
               confident.
    """

    P: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    omega: npt.NDArray[np.float64]


class BlackLittermanViews:
    """Converts RL agent Q-value signals into Bayesian Black-Litterman views.

    The idea: the Q-value gap between LONG and SHORT actions provides a
    forward-looking return expectation.  This is normalised and injected as
    a relative view.

    Args:
        assets: Ordered list of asset ticker symbols.
        tau: Uncertainty scaling of the prior (default 0.05).
        view_confidence: Default confidence (inverse of omega) for each view.
    """

    def __init__(
        self,
        assets: Sequence[str],
        tau: float = 0.05,
        view_confidence: float = 0.5,
    ) -> None:
        self._assets = list(assets)
        self._n = len(assets)
        self._tau = tau
        self._view_confidence = view_confidence

    def from_q_values(
        self,
        q_long: npt.NDArray[np.float64],
        q_short: npt.NDArray[np.float64],
    ) -> BLViewSet:
        """Build a :class:`BLViewSet` from per-asset Q-value differences.

        A positive Q-gap (long Q > short Q) translates to a bullish view;
        negative means bearish.  Views are expressed as absolute return
        expectations relative to a zero baseline.

        Args:
            q_long: Q-values for the LONG action, shape ``(N,)``.
            q_short: Q-values for the SHORT action, shape ``(N,)``.

        Returns:
            :class:`BLViewSet` with one view per asset.

        Raises:
            DataFetchError: If array shapes are incompatible.
        """
        q_long = np.asarray(q_long, dtype=np.float64).flatten()
        q_short = np.asarray(q_short, dtype=np.float64).flatten()

        if q_long.shape != (self._n,) or q_short.shape != (self._n,):
            raise DataFetchError(
                f"q_long and q_short must each have shape ({self._n},)."
            )

        gap = q_long - q_short
        # Normalise gaps to plausible return expectations (e.g. 1% range)
        max_gap = np.max(np.abs(gap)) + _EPSILON
        Q = gap / max_gap * 0.01  # scale to ±1% expected return

        P = np.eye(self._n)
        omega_val = (1.0 - self._view_confidence) / self._view_confidence
        omega = np.full(self._n, omega_val)

        log.debug(
            "bl_views.built",
            assets=self._assets,
            Q_min=round(float(Q.min()), 5),
            Q_max=round(float(Q.max()), 5),
        )
        return BLViewSet(P=P, Q=Q, omega=omega)

    def posterior_returns(
        self,
        prior_returns: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64],
        views: BLViewSet,
    ) -> npt.NDArray[np.float64]:
        """Compute the Black-Litterman posterior expected returns.

        Implements the standard BL update:
            mu_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹ π + P'Ω⁻¹Q]

        Args:
            prior_returns: Market-implied equilibrium returns, shape ``(N,)``.
            cov: Asset covariance matrix, shape ``(N, N)``.
            views: :class:`BLViewSet` with P, Q, omega.

        Returns:
            Posterior expected returns, shape ``(N,)``.
        """
        tau_cov = self._tau * cov
        try:
            tau_cov_inv = np.linalg.inv(tau_cov)
        except np.linalg.LinAlgError:
            tau_cov_inv = np.linalg.pinv(tau_cov)

        P, Q, omega = views.P, views.Q, views.omega
        omega_inv = np.diag(1.0 / (omega + _EPSILON))

        A = tau_cov_inv + P.T @ omega_inv @ P
        b = tau_cov_inv @ prior_returns + P.T @ omega_inv @ Q

        try:
            mu_bl = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            mu_bl = np.linalg.lstsq(A, b, rcond=None)[0]

        log.debug("bl_views.posterior", mu_bl=mu_bl.round(5).tolist())
        return mu_bl


# ---------------------------------------------------------------------------
# Mean-Variance Optimizer (Markowitz + optional BL)
# ---------------------------------------------------------------------------


class MeanVarianceOptimizer:
    """Classic Markowitz mean-variance optimiser with optional BL views.

    Maximises the Sharpe ratio subject to long-only and full-investment
    constraints.  When *bl_views* is provided, the Black-Litterman posterior
    replaces the sample mean as the expected return vector.

    Args:
        assets: Ordered list of asset ticker symbols.
        risk_free_rate: Annualised risk-free rate (default 0.0).
        bl: Optional pre-configured :class:`BlackLittermanViews`.
    """

    def __init__(
        self,
        assets: Sequence[str],
        risk_free_rate: float = 0.0,
        bl: BlackLittermanViews | None = None,
    ) -> None:
        self._assets = list(assets)
        self._n = len(assets)
        self._rf = risk_free_rate
        self._bl = bl

    def allocate(
        self,
        returns: npt.NDArray[np.float32],
        bl_views: BLViewSet | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute maximum-Sharpe weights.

        Args:
            returns: Historical return matrix of shape ``(T, N)``.
            bl_views: Optional :class:`BLViewSet` to inject as Black-Litterman
                views.  Requires that *bl* was set in the constructor.

        Returns:
            Weight vector of shape ``(N,)`` summing to 1.0.

        Raises:
            DataFetchError: If *returns* is malformed.
            ModelError: If the optimisation fails.
        """
        _validate_returns(returns, self._n)

        cov = _cov_shrinkage(returns)
        mu = np.mean(returns, axis=0).astype(np.float64)

        # Apply Black-Litterman if views provided
        if bl_views is not None and self._bl is not None:
            mu = self._bl.posterior_returns(
                prior_returns=mu,
                cov=cov,
                views=bl_views,
            )

        def neg_sharpe(w: npt.NDArray) -> float:
            port_ret = float(np.dot(w, mu))
            port_vol = float(np.sqrt(np.dot(w, cov @ w) + _EPSILON))
            return -(port_ret - self._rf) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * self._n
        w0 = np.full(self._n, 1.0 / self._n)

        try:
            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            weights = _normalise(np.clip(result.x, 0.0, 1.0))
        except Exception as exc:
            raise ModelError("Mean-variance optimisation failed.", cause=exc) from exc

        log.info(
            "mv_optimizer.allocated",
            assets=self._assets,
            weights=weights.round(4).tolist(),
            sharpe=round(-neg_sharpe(weights), 4),
        )
        return weights


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------


class HierarchicalRiskParity:
    """Hierarchical Risk Parity allocation (López de Prado, 2016).

    Clusters assets by correlation using Ward linkage, then recursively
    splits the dendrogram and equalises risk contribution within each cluster.

    Args:
        assets: Ordered list of asset ticker symbols.
        linkage_method: Scipy linkage method (default "ward").
    """

    def __init__(
        self,
        assets: Sequence[str],
        linkage_method: str = "ward",
    ) -> None:
        self._assets = list(assets)
        self._n = len(assets)
        self._linkage_method = linkage_method

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_distance(corr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a correlation matrix to a distance matrix sqrt((1-rho)/2)."""
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _get_ivp(cov: npt.NDArray[np.float64], indices: list[int]) -> npt.NDArray[np.float64]:
        """Inverse-variance portfolio over the sub-set of *indices*."""
        sub_cov = cov[np.ix_(indices, indices)]
        ivp = 1.0 / (np.diag(sub_cov) + _EPSILON)
        return ivp / ivp.sum()

    def _hrp_weights(
        self,
        cov: npt.NDArray[np.float64],
        sorted_indices: list[int],
    ) -> npt.NDArray[np.float64]:
        """Recursive bisection of the quasi-diagonalised covariance matrix."""
        weights = np.ones(self._n)

        def _recurse(items: list[int]) -> None:
            if len(items) <= 1:
                return
            mid = len(items) // 2
            left, right = items[:mid], items[mid:]

            var_left = float(
                np.dot(
                    self._get_ivp(cov, left),
                    (cov[np.ix_(left, left)] @ self._get_ivp(cov, left)),
                )
            )
            var_right = float(
                np.dot(
                    self._get_ivp(cov, right),
                    (cov[np.ix_(right, right)] @ self._get_ivp(cov, right)),
                )
            )
            alloc_left = var_right / (var_left + var_right + _EPSILON)
            alloc_right = 1.0 - alloc_left

            weights[left] *= alloc_left
            weights[right] *= alloc_right

            _recurse(left)
            _recurse(right)

        _recurse(sorted_indices)
        return _normalise(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, returns: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Compute HRP weights from historical returns.

        Args:
            returns: Return matrix of shape ``(T, N)``.

        Returns:
            Weight vector of shape ``(N,)`` summing to 1.0.

        Raises:
            DataFetchError: If *returns* is malformed.
        """
        _validate_returns(returns, self._n)

        cov = _cov_shrinkage(returns)
        corr = np.corrcoef(returns.T)

        dist_mat = self._correlation_distance(corr)
        condensed = squareform(dist_mat)
        link = linkage(condensed, method=self._linkage_method)

        # Quasi-diagonalise: obtain the leaf order from the dendrogram
        from scipy.cluster.hierarchy import leaves_list

        sorted_idx = list(leaves_list(link))
        weights = self._hrp_weights(cov, sorted_idx)

        log.info(
            "hrp.allocated",
            assets=self._assets,
            weights=weights.round(4).tolist(),
        )
        return weights


# ---------------------------------------------------------------------------
# Minimum Correlation Portfolio
# ---------------------------------------------------------------------------


class MinimumCorrelation:
    """Minimises the weighted average pairwise correlation of the portfolio.

    Objective:
        min  w^T C w    (C = correlation matrix)
        s.t. sum(w) = 1,  w >= 0

    Args:
        assets: Ordered list of asset ticker symbols.
    """

    def __init__(self, assets: Sequence[str]) -> None:
        self._assets = list(assets)
        self._n = len(assets)

    def allocate(self, returns: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Compute minimum-correlation weights.

        Args:
            returns: Return matrix of shape ``(T, N)``.

        Returns:
            Weight vector of shape ``(N,)`` summing to 1.0.

        Raises:
            DataFetchError: If *returns* is malformed.
            ModelError: If the optimisation fails.
        """
        _validate_returns(returns, self._n)

        corr = np.corrcoef(returns.T)

        def portfolio_correlation(w: npt.NDArray) -> float:
            return float(w @ corr @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * self._n
        w0 = np.full(self._n, 1.0 / self._n)

        try:
            result = minimize(
                portfolio_correlation,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            weights = _normalise(np.clip(result.x, 0.0, 1.0))
        except Exception as exc:
            raise ModelError(
                "Minimum-correlation optimisation failed.", cause=exc
            ) from exc

        avg_corr = float(weights @ corr @ weights)
        log.info(
            "min_corr.allocated",
            assets=self._assets,
            weights=weights.round(4).tolist(),
            avg_corr=round(avg_corr, 4),
        )
        return weights


# ---------------------------------------------------------------------------
# Portfolio Rebalancer
# ---------------------------------------------------------------------------


class OptimStrategy(str, Enum):
    """Portfolio optimisation strategy selector."""

    MEAN_VARIANCE = "mean_variance"
    HRP = "hrp"
    MIN_CORRELATION = "min_correlation"


@dataclass
class RebalanceRecord:
    """Record of a single rebalancing event.

    Attributes:
        timestamp: Wall-clock time of the rebalance.
        old_weights: Portfolio weights before rebalancing.
        new_weights: Target weights after optimisation.
        trades: Per-asset signed weight deltas.
        estimated_cost: Estimated transaction cost as a fraction of portfolio.
        strategy: The optimisation strategy used.
    """

    timestamp: float
    old_weights: npt.NDArray[np.float64]
    new_weights: npt.NDArray[np.float64]
    trades: npt.NDArray[np.float64]
    estimated_cost: float
    strategy: OptimStrategy


class PortfolioRebalancer:
    """Periodic portfolio rebalancer with transaction-cost awareness.

    Computes target weights via the selected strategy and only executes a
    rebalance when the total weight deviation exceeds *drift_threshold* AND
    the estimated transaction cost is below *max_cost_fraction*.

    Args:
        assets: Ordered list of asset ticker symbols.
        strategy: Optimisation method to use.
        drift_threshold: Minimum total weight drift (L1 norm) to trigger a
            rebalance (default 0.05 = 5%).
        transaction_cost_bps: Estimated round-trip transaction cost in basis
            points (default 10 bps = 0.10%).
        max_cost_fraction: Skip the rebalance if estimated cost exceeds this
            fraction of portfolio value (default 0.005 = 0.5%).
        lookback: Number of return observations used for estimation.
        bl: Optional :class:`BlackLittermanViews` for mean-variance mode.
    """

    def __init__(
        self,
        assets: Sequence[str],
        strategy: OptimStrategy = OptimStrategy.HRP,
        drift_threshold: float = 0.05,
        transaction_cost_bps: float = 10.0,
        max_cost_fraction: float = 0.005,
        lookback: int = 120,
        bl: BlackLittermanViews | None = None,
    ) -> None:
        self._assets = list(assets)
        self._n = len(assets)
        self._strategy = strategy
        self._drift_threshold = drift_threshold
        self._tc_bps = transaction_cost_bps
        self._max_cost = max_cost_fraction
        self._lookback = lookback

        # Instantiate all optimisers so switching strategy at runtime is cheap
        self._mv = MeanVarianceOptimizer(assets=assets, bl=bl)
        self._hrp = HierarchicalRiskParity(assets=assets)
        self._mc = MinimumCorrelation(assets=assets)

        self._current_weights: npt.NDArray[np.float64] = np.full(
            self._n, 1.0 / self._n
        )
        self._history: list[RebalanceRecord] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _optimise(
        self,
        returns: npt.NDArray[np.float32],
        bl_views: BLViewSet | None = None,
    ) -> npt.NDArray[np.float64]:
        """Run the selected optimiser."""
        if self._strategy == OptimStrategy.MEAN_VARIANCE:
            return self._mv.allocate(returns, bl_views=bl_views)
        if self._strategy == OptimStrategy.HRP:
            return self._hrp.allocate(returns)
        if self._strategy == OptimStrategy.MIN_CORRELATION:
            return self._mc.allocate(returns)
        raise ModelError(f"Unknown strategy: {self._strategy}")  # type: ignore[arg-type]

    def _estimate_cost(self, trades: npt.NDArray[np.float64]) -> float:
        """Estimate round-trip transaction cost from weight deltas.

        Args:
            trades: Signed weight changes, shape ``(N,)``.

        Returns:
            Estimated cost as a fraction of total portfolio value.
        """
        turnover = float(np.sum(np.abs(trades))) / 2.0
        return turnover * self._tc_bps / 10_000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebalance(
        self,
        returns: npt.NDArray[np.float32],
        bl_views: BLViewSet | None = None,
        force: bool = False,
    ) -> RebalanceRecord | None:
        """Attempt a portfolio rebalance.

        Computes target weights and executes the rebalance only when:
        - Total weight drift exceeds *drift_threshold*, OR *force* is True.
        - Estimated transaction cost is below *max_cost_fraction*.

        Args:
            returns: Recent return matrix of shape ``(T, N)``.  Only the last
                *lookback* rows are used.
            bl_views: Optional Black-Litterman views for mean-variance strategy.
            force: If True, skip the drift check and always rebalance.

        Returns:
            A :class:`RebalanceRecord` if a rebalance was executed, or ``None``
            if the rebalance was skipped.

        Raises:
            DataFetchError: If *returns* is malformed.
        """
        # Use only the most recent lookback window
        recent = returns[-self._lookback :].astype(np.float32)
        _validate_returns(recent, self._n)

        target = self._optimise(recent, bl_views=bl_views)
        trades = target - self._current_weights
        drift = float(np.sum(np.abs(trades)))
        cost = self._estimate_cost(trades)

        if not force and drift < self._drift_threshold:
            log.debug(
                "rebalancer.skip_drift",
                drift=round(drift, 4),
                threshold=self._drift_threshold,
            )
            return None

        if cost > self._max_cost:
            log.warning(
                "rebalancer.skip_cost",
                cost=round(cost, 6),
                max_cost=self._max_cost,
            )
            return None

        record = RebalanceRecord(
            timestamp=time.time(),
            old_weights=self._current_weights.copy(),
            new_weights=target.copy(),
            trades=trades.copy(),
            estimated_cost=cost,
            strategy=self._strategy,
        )
        self._history.append(record)
        self._current_weights = target

        log.info(
            "rebalancer.executed",
            strategy=self._strategy.value,
            drift=round(drift, 4),
            cost_bps=round(cost * 10_000, 2),
            old_weights=record.old_weights.round(4).tolist(),
            new_weights=target.round(4).tolist(),
        )
        return record

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_weights(self) -> npt.NDArray[np.float64]:
        """Current portfolio weight vector (copy)."""
        return self._current_weights.copy()

    @property
    def history(self) -> list[RebalanceRecord]:
        """Ordered list of all rebalance events."""
        return list(self._history)

    @property
    def strategy(self) -> OptimStrategy:
        """Currently selected optimisation strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: OptimStrategy) -> None:
        """Switch to a different optimisation strategy at runtime."""
        log.info("rebalancer.strategy_changed", old=self._strategy.value, new=value.value)
        self._strategy = value

    def total_cost(self) -> float:
        """Cumulative estimated transaction costs across all rebalances."""
        return sum(r.estimated_cost for r in self._history)

    def summary(self) -> str:
        """Return a multi-line summary of portfolio state.

        Returns:
            Human-readable string with current weights, strategy, and history.
        """
        lines = [
            "=== Portfolio Rebalancer ===",
            f"Strategy      : {self._strategy.value}",
            f"Assets        : {self._assets}",
            f"Current wts   : {self._current_weights.round(4).tolist()}",
            f"Rebalances    : {len(self._history)}",
            f"Total est cost: {self.total_cost() * 10_000:.2f} bps",
        ]
        if self._history:
            last = self._history[-1]
            lines += [
                "",
                "Last rebalance:",
                f"  Old weights : {last.old_weights.round(4).tolist()}",
                f"  New weights : {last.new_weights.round(4).tolist()}",
                f"  Est. cost   : {last.estimated_cost * 10_000:.2f} bps",
            ]
        return "\n".join(lines)

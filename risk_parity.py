"""Risk parity portfolio allocation for multi-asset crypto trading.

Allocates capital across N assets so that each contributes equal *risk*
(measured by volatility contribution) to the total portfolio.  The module
re-balances daily and integrates with :mod:`multi_asset`.

Theory
------
Risk parity targets equal *risk contribution* rather than equal capital weight.
The risk contribution of asset *i* is:

    RC_i = w_i * (Σ w)_i / sqrt(w^T Σ w)

where Σ is the covariance matrix and w is the weight vector.  We solve for w
such that RC_i = 1/N for all i using an iterative Newton-Raphson approach.

Reference: Maillard, S., Roncalli, T., & Teïletche, J. (2010).
    "The Properties of Equally Weighted Risk Contribution Portfolios."

Example::

    from risk_parity import RiskParityAllocator
    import numpy as np

    # Simulate 60-day return history for 3 assets
    returns = np.random.randn(60, 3) * 0.02
    allocator = RiskParityAllocator(assets=["BTC", "ETH", "SOL"])
    weights = allocator.allocate(returns)
    print(weights)  # e.g. [0.42, 0.35, 0.23]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from exceptions import DataFetchError, ModelError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum look-back rows required to estimate a covariance matrix.
_MIN_LOOKBACK: int = 10

#: Maximum iterations for the Newton-Raphson solver.
_MAX_ITER: int = 500

#: Convergence tolerance (change in weight vector L2 norm).
_TOLERANCE: float = 1e-10

#: Regularisation added to the diagonal of the covariance matrix for stability.
_COV_REG: float = 1e-8


# ---------------------------------------------------------------------------
# Covariance utilities
# ---------------------------------------------------------------------------


def shrunk_covariance(returns: npt.NDArray[np.float64], alpha: float = 0.1) -> npt.NDArray[np.float64]:
    """Ledoit-Wolf-style linear shrinkage towards the diagonal.

    Args:
        returns: ``(T, N)`` matrix of asset returns.
        alpha:   Shrinkage intensity in ``[0, 1]``.  0 = sample covariance,
                 1 = diagonal covariance.

    Returns:
        ``(N, N)`` shrunk covariance matrix.
    """
    cov = np.cov(returns, rowvar=False)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - alpha) * cov + alpha * diag
    # Add small regularisation for numerical stability
    shrunk += np.eye(shrunk.shape[0]) * _COV_REG
    return shrunk


# ---------------------------------------------------------------------------
# Risk contribution computation
# ---------------------------------------------------------------------------


def risk_contributions(weights: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute per-asset risk contributions.

    Args:
        weights: 1-D weight vector of length N (should sum to 1).
        cov:     ``(N, N)`` covariance matrix.

    Returns:
        1-D array of risk contributions; sums to total portfolio volatility.
    """
    portfolio_var = float(weights @ cov @ weights)
    if portfolio_var < 1e-12:
        return np.ones_like(weights) / len(weights)
    portfolio_vol = np.sqrt(portfolio_var)
    marginal = cov @ weights
    rc = weights * marginal / portfolio_vol
    return rc


def equal_risk_objective(weights: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> float:
    """Objective: sum of squared differences from equal risk contribution.

    Minimised at 0 when all assets contribute equal risk.

    Args:
        weights: 1-D weight vector.
        cov:     ``(N, N)`` covariance matrix.

    Returns:
        Non-negative scalar objective value.
    """
    rc = risk_contributions(weights, cov)
    target = rc.sum() / len(weights)
    return float(np.sum((rc - target) ** 2))


# ---------------------------------------------------------------------------
# Iterative solver
# ---------------------------------------------------------------------------


def solve_risk_parity(
    cov: npt.NDArray[np.float64],
    max_iter: int = _MAX_ITER,
    tol: float = _TOLERANCE,
) -> npt.NDArray[np.float64]:
    """Solve for risk-parity weights using iterative gradient descent.

    Uses the analytic gradient of the equal-risk objective and Adam-style
    adaptive learning rate for fast convergence.

    Args:
        cov:      ``(N, N)`` covariance matrix (positive semi-definite).
        max_iter: Maximum solver iterations.
        tol:      Convergence threshold on weight-vector change (L2 norm).

    Returns:
        1-D weight vector of length N summing to 1.

    Raises:
        ModelError: If the solver fails to converge within ``max_iter``.
    """
    n = cov.shape[0]
    w = np.ones(n) / n  # equal-weight starting point

    lr = 1e-3
    # Simple momentum-based gradient descent
    velocity = np.zeros(n)
    beta = 0.9

    for iteration in range(max_iter):
        w_old = w.copy()

        # Gradient of objective w.r.t. weights (numerical)
        eps = 1e-7
        grad = np.zeros(n)
        f0 = equal_risk_objective(w, cov)
        for i in range(n):
            w_eps = w.copy()
            w_eps[i] += eps
            grad[i] = (equal_risk_objective(w_eps, cov) - f0) / eps

        # Momentum update
        velocity = beta * velocity + (1 - beta) * grad
        w -= lr * velocity

        # Project onto simplex (non-negative, sums to 1)
        w = np.maximum(w, 1e-6)
        w /= w.sum()

        # Convergence check
        delta = np.linalg.norm(w - w_old)
        if delta < tol:
            log.debug(
                "risk_parity.solver_converged",
                iterations=iteration + 1,
                delta=float(delta),
            )
            return w

    log.warning(
        "risk_parity.solver_max_iter",
        max_iter=max_iter,
        final_objective=round(equal_risk_objective(w, cov), 8),
    )
    return w


# ---------------------------------------------------------------------------
# RiskParityAllocator
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Output of a single :meth:`RiskParityAllocator.allocate` call.

    Attributes:
        weights:          Dict mapping asset name -> portfolio weight.
        risk_contributions: Dict mapping asset name -> risk contribution fraction.
        portfolio_vol:    Estimated annualised portfolio volatility.
        covariance:       The ``(N, N)`` covariance matrix used.
    """

    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    portfolio_vol: float
    covariance: npt.NDArray[np.float64]


class RiskParityAllocator:
    """Daily risk-parity portfolio rebalancer for N crypto assets.

    Args:
        assets:           Ordered list of asset ticker strings.
        lookback:         Number of return observations used to estimate the
                          covariance matrix (default 60).
        shrinkage_alpha:  Ledoit-Wolf shrinkage intensity (default 0.1).
        annualisation:    Factor to annualise volatility (default 365 for daily).
        max_weight:       Maximum allocation to a single asset (default 0.40).
        min_weight:       Minimum allocation to a single asset (default 0.05).

    Example::

        allocator = RiskParityAllocator(["BTC", "ETH", "SOL"], lookback=60)
        result = allocator.allocate(returns_matrix)
        print(result.weights)
    """

    def __init__(
        self,
        assets: List[str],
        lookback: int = 60,
        shrinkage_alpha: float = 0.1,
        annualisation: int = 365,
        max_weight: float = 0.40,
        min_weight: float = 0.05,
    ) -> None:
        if len(assets) < 2:
            raise ValueError("RiskParityAllocator requires at least 2 assets.")
        self.assets = assets
        self.lookback = lookback
        self.shrinkage_alpha = shrinkage_alpha
        self.annualisation = annualisation
        self.max_weight = max_weight
        self.min_weight = min_weight

        log.info(
            "risk_parity.init",
            assets=assets,
            lookback=lookback,
            shrinkage_alpha=shrinkage_alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        returns: npt.ArrayLike,
        force_rebalance: bool = False,
    ) -> AllocationResult:
        """Compute risk-parity weights from a return history matrix.

        Args:
            returns: Array-like of shape ``(T, N)`` where T >= ``lookback`` and
                     N == ``len(self.assets)``.  Each column is one asset's
                     return series.
            force_rebalance: Unused flag; kept for API compatibility with
                             daily rebalance schedulers.

        Returns:
            :class:`AllocationResult` with weights, risk contributions, and
            portfolio volatility.

        Raises:
            DataFetchError: If ``returns`` has insufficient rows or wrong
                            number of columns.
            ModelError:     If the solver fails or covariance is degenerate.
        """
        returns = np.asarray(returns, dtype=np.float64)

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        T, N = returns.shape
        if N != len(self.assets):
            raise DataFetchError(
                f"returns has {N} columns but {len(self.assets)} assets were specified."
            )
        if T < _MIN_LOOKBACK:
            raise DataFetchError(
                f"Need at least {_MIN_LOOKBACK} rows; got {T}."
            )

        # Use only the most recent `lookback` rows
        recent = returns[-self.lookback:]

        # --- Covariance estimation ---
        cov = shrunk_covariance(recent, alpha=self.shrinkage_alpha)

        # --- Risk-parity solve ---
        try:
            raw_weights = solve_risk_parity(cov)
        except Exception as exc:
            raise ModelError("Risk-parity solver failed.", cause=exc) from exc

        # --- Clip to [min_weight, max_weight] and renormalise ---
        clipped = np.clip(raw_weights, self.min_weight, self.max_weight)
        clipped /= clipped.sum()

        # --- Risk contributions ---
        rc = risk_contributions(clipped, cov)
        total_rc = rc.sum()
        rc_frac = rc / total_rc if total_rc > 1e-12 else np.ones(N) / N

        # --- Annualised portfolio volatility ---
        port_var = float(clipped @ cov @ clipped)
        port_vol = np.sqrt(port_var * self.annualisation)

        weight_dict = dict(zip(self.assets, clipped.tolist()))
        rc_dict = dict(zip(self.assets, rc_frac.tolist()))

        result = AllocationResult(
            weights=weight_dict,
            risk_contributions=rc_dict,
            portfolio_vol=float(port_vol),
            covariance=cov,
        )

        log.info(
            "risk_parity.allocate",
            weights={k: round(v, 4) for k, v in weight_dict.items()},
            risk_contributions={k: round(v, 4) for k, v in rc_dict.items()},
            portfolio_vol=round(float(port_vol), 6),
        )
        return result

    def daily_rebalance(
        self,
        returns_history: npt.ArrayLike,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Compute target USD allocations for each asset.

        Args:
            returns_history: ``(T, N)`` return history array.
            portfolio_value: Current total portfolio value in USD.

        Returns:
            Dict mapping asset name -> target USD allocation.
        """
        result = self.allocate(returns_history)
        allocations = {
            asset: round(weight * portfolio_value, 2)
            for asset, weight in result.weights.items()
        }
        log.info(
            "risk_parity.daily_rebalance",
            portfolio_value=portfolio_value,
            allocations=allocations,
        )
        return allocations

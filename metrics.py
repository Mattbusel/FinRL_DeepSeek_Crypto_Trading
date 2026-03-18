"""Financial performance metrics for the LARSA evaluation pipeline.

All functions accept a sequence of percentage returns (i.e. period-over-period
relative changes) and return a scalar metric.

Example::

    from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
    sr = sharpe_ratio(returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import empyrical as ep

from exceptions import SignalError


def cumulative_returns(returns_pct: np.ndarray | list[float]) -> pd.Series:
    """Compute the cumulative return series from percentage returns.

    Args:
        returns_pct: Sequence of per-step percentage returns,
            e.g. ``(r_{t+1} - r_t) / r_t``.

    Returns:
        A :class:`pandas.Series` of cumulative returns.

    Raises:
        SignalError: If *returns_pct* is empty.
    """
    arr = np.asarray(returns_pct, dtype=float)
    if arr.size == 0:
        raise SignalError("cumulative_returns received an empty returns array.")
    return ep.cum_returns(arr)


def sharpe_ratio(
    returns_pct: np.ndarray | list[float], risk_free: float = 0.0
) -> float:
    """Compute the annualisation-free Sharpe ratio.

    Uses the sample standard deviation (``ddof=1``).  Returns ``inf`` when the
    standard deviation is zero (i.e. all returns are identical).

    Args:
        returns_pct: Sequence of per-step percentage returns.
        risk_free: Risk-free rate to subtract from the mean return (default 0).

    Returns:
        The Sharpe ratio as a float, or ``inf`` if std is zero.

    Raises:
        SignalError: If *returns_pct* is empty.
    """
    arr = np.asarray(returns_pct, dtype=float)
    if arr.size == 0:
        raise SignalError("sharpe_ratio received an empty returns array.")
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return float(np.inf)
    return float((arr.mean() - risk_free) / std)


def max_drawdown(returns_pct: np.ndarray | list[float]) -> float:
    """Compute the maximum drawdown from a percentage returns sequence.

    Args:
        returns_pct: Sequence of per-step percentage returns.

    Returns:
        Maximum drawdown as a non-positive float (e.g. ``-0.35`` means -35%).

    Raises:
        SignalError: If *returns_pct* is empty.
    """
    arr = np.asarray(returns_pct, dtype=float)
    if arr.size == 0:
        raise SignalError("max_drawdown received an empty returns array.")
    return float(ep.max_drawdown(arr))


def return_over_max_drawdown(returns_pct: np.ndarray | list[float]) -> float:
    """Compute the Calmar-style return-over-max-drawdown (RoMaD) ratio.

    Returns ``inf`` when max drawdown is zero (no losing streak observed).

    Args:
        returns_pct: Sequence of per-step percentage returns.

    Returns:
        Ratio of cumulative return to absolute max drawdown, or ``inf`` if
        max drawdown is zero.

    Raises:
        SignalError: If *returns_pct* is empty.
    """
    mdd = abs(max_drawdown(returns_pct))
    cum_ret = float(cumulative_returns(returns_pct).iloc[-1])
    if mdd == 0.0:
        return float(np.inf)
    return cum_ret / mdd

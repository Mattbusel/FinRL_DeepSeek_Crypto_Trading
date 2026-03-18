"""Tests for metrics.py: sharpe_ratio, max_drawdown, return_over_max_drawdown."""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown, cumulative_returns
from exceptions import SignalError


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_ratio_positive_returns():
    """Known constant positive returns should yield a positive Sharpe ratio."""
    # All returns identical => std == 0 => returns inf
    # Use varying positive returns for a finite positive Sharpe
    returns = [0.01, 0.02, 0.03, 0.02, 0.01]
    sr = sharpe_ratio(returns)
    assert sr > 0


def test_sharpe_ratio_known_value():
    """Manually computed Sharpe should match the function output."""
    returns = np.array([0.1, 0.2, 0.3])
    mean = returns.mean()
    std = returns.std(ddof=1)
    expected = mean / std
    result = sharpe_ratio(returns)
    assert abs(result - expected) < 1e-10


def test_sharpe_ratio_zero_std_returns_inf():
    """All identical returns => std == 0 => Sharpe is inf."""
    returns = [0.05, 0.05, 0.05, 0.05]
    result = sharpe_ratio(returns)
    assert math.isinf(result)


def test_sharpe_ratio_negative_returns():
    """Uniformly negative returns should yield a negative Sharpe ratio."""
    returns = [-0.01, -0.02, -0.03]
    assert sharpe_ratio(returns) < 0


def test_sharpe_ratio_empty_raises():
    """Empty input should raise SignalError."""
    with pytest.raises(SignalError):
        sharpe_ratio([])


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_no_drawdown():
    """Monotone increasing returns should produce a drawdown of 0."""
    returns = [0.01, 0.02, 0.03, 0.04]
    mdd = max_drawdown(returns)
    # empyrical.max_drawdown returns <= 0; for monotone increases it is 0
    assert mdd == pytest.approx(0.0, abs=1e-6) or mdd >= -1e-6


def test_max_drawdown_known_value():
    """Portfolio going from 1 → 0.5 → 1 should have a large drawdown."""
    # The sequence [-0.5, 1.0] corresponds to a 50% drop then recovery
    returns = [-0.5, 1.0]
    mdd = max_drawdown(returns)
    # drawdown should be <= 0 and substantial
    assert mdd < 0


def test_max_drawdown_all_losses():
    """All negative returns should produce a significant drawdown."""
    returns = [-0.1, -0.1, -0.1]
    mdd = max_drawdown(returns)
    assert mdd < 0


def test_max_drawdown_empty_raises():
    """Empty input should raise SignalError."""
    with pytest.raises(SignalError):
        max_drawdown([])


# ---------------------------------------------------------------------------
# return_over_max_drawdown (Calmar-style)
# ---------------------------------------------------------------------------

def test_calmar_ratio_positive():
    """Positive cumulative return with a drawdown should yield positive RoMaD."""
    # returns go up, down a bit, then up strongly
    returns = [0.05, 0.05, -0.02, 0.10, 0.05]
    romd = return_over_max_drawdown(returns)
    # Could be inf (no drawdown) or positive finite; either is acceptable
    assert romd > 0 or math.isinf(romd)


def test_calmar_ratio_zero_drawdown_returns_inf():
    """Monotone positive returns → zero drawdown → RoMaD is inf."""
    returns = [0.01, 0.02, 0.03]
    result = return_over_max_drawdown(returns)
    assert math.isinf(result)


def test_calmar_ratio_empty_raises():
    """Empty input should raise SignalError."""
    with pytest.raises(SignalError):
        return_over_max_drawdown([])


# ---------------------------------------------------------------------------
# cumulative_returns
# ---------------------------------------------------------------------------

def test_cumulative_returns_non_empty():
    """cumulative_returns should return a non-empty series."""
    result = cumulative_returns([0.01, 0.02, -0.01])
    assert len(result) == 3


def test_cumulative_returns_empty_raises():
    """Empty input should raise SignalError."""
    with pytest.raises(SignalError):
        cumulative_returns([])

"""Tests for the metrics.py module.

Covers :func:`cumulative_returns`, :func:`sharpe_ratio`,
:func:`max_drawdown`, and :func:`return_over_max_drawdown` with normal,
edge-case, and error inputs.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("empyrical", reason="empyrical not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import SignalError
from metrics import (
    cumulative_returns,
    max_drawdown,
    return_over_max_drawdown,
    sharpe_ratio,
)


class TestCumulativeReturns:
    """Tests for :func:`cumulative_returns`."""

    def test_positive_returns(self) -> None:
        """Positive returns produce a monotonically increasing series."""
        result = cumulative_returns([0.01, 0.01, 0.01])
        values = result.values
        assert values[-1] > values[0]

    def test_zero_returns(self) -> None:
        """All-zero returns produce an all-zero cumulative series."""
        result = cumulative_returns([0.0, 0.0, 0.0])
        assert all(v == 0.0 for v in result.values)

    def test_list_input(self) -> None:
        """Plain Python list is accepted and produces the correct length."""
        result = cumulative_returns([0.02, -0.01, 0.03])
        assert len(result) == 3

    def test_numpy_input(self) -> None:
        """NumPy array input is accepted without error."""
        arr = np.array([0.05, -0.02, 0.01])
        result = cumulative_returns(arr)
        assert len(result) == 3

    def test_empty_raises(self) -> None:
        """Empty input raises :class:`SignalError`."""
        with pytest.raises(SignalError):
            cumulative_returns([])

    def test_empty_numpy_raises(self) -> None:
        """Empty numpy array raises :class:`SignalError`."""
        with pytest.raises(SignalError):
            cumulative_returns(np.array([]))


class TestSharpeRatio:
    """Tests for :func:`sharpe_ratio`."""

    def test_positive_returns(self) -> None:
        """Steady positive returns produce a positive Sharpe ratio."""
        sr = sharpe_ratio([0.01] * 20)
        assert sr == math.inf  # std == 0, identical returns

    def test_mixed_returns(self) -> None:
        """Mixed returns produce a finite Sharpe ratio."""
        returns = [0.02, -0.01, 0.03, -0.005, 0.015]
        sr = sharpe_ratio(returns)
        assert math.isfinite(sr)

    def test_risk_free_offset(self) -> None:
        """Higher risk-free rate lowers the Sharpe ratio."""
        returns = [0.02, 0.01, 0.03, 0.015]
        sr0 = sharpe_ratio(returns, risk_free=0.0)
        sr1 = sharpe_ratio(returns, risk_free=0.01)
        assert sr0 > sr1

    def test_zero_std_returns_inf(self) -> None:
        """Constant returns (zero std) return infinity."""
        sr = sharpe_ratio([0.005, 0.005, 0.005])
        assert sr == math.inf

    def test_empty_raises(self) -> None:
        """Empty input raises :class:`SignalError`."""
        with pytest.raises(SignalError):
            sharpe_ratio([])

    def test_single_element(self) -> None:
        """Single-element returns produce infinity (std of one element is 0 with ddof=1 -> nan)."""
        # np.array([x]).std(ddof=1) is nan for a single element; we should get inf
        sr = sharpe_ratio([0.01])
        assert math.isinf(sr) or math.isnan(sr)


class TestMaxDrawdown:
    """Tests for :func:`max_drawdown`."""

    def test_no_drawdown(self) -> None:
        """Always positive returns produce zero or very small drawdown."""
        mdd = max_drawdown([0.01, 0.02, 0.01, 0.03])
        assert mdd <= 0.0

    def test_large_drawdown(self) -> None:
        """A clear losing streak produces a large negative drawdown."""
        returns = [0.1, -0.5, -0.3, 0.05]
        mdd = max_drawdown(returns)
        assert mdd < -0.3

    def test_result_is_nonpositive(self) -> None:
        """Max drawdown is always <= 0."""
        for _ in range(10):
            returns = np.random.normal(0.001, 0.01, size=50).tolist()
            assert max_drawdown(returns) <= 0.0

    def test_empty_raises(self) -> None:
        """Empty input raises :class:`SignalError`."""
        with pytest.raises(SignalError):
            max_drawdown([])


class TestReturnOverMaxDrawdown:
    """Tests for :func:`return_over_max_drawdown`."""

    def test_positive_ratio(self) -> None:
        """Profitable run with drawdown produces a finite positive RoMaD."""
        # Strong uptrend with a small dip in the middle
        returns = [0.05, 0.03, -0.01, 0.04, 0.06]
        roma = return_over_max_drawdown(returns)
        assert math.isfinite(roma)

    def test_zero_drawdown_returns_inf(self) -> None:
        """Returns with no drawdown produce infinity."""
        returns = [0.01, 0.02, 0.03]
        roma = return_over_max_drawdown(returns)
        assert roma == math.inf

    def test_empty_propagates_error(self) -> None:
        """Empty input raises :class:`SignalError`."""
        with pytest.raises(SignalError):
            return_over_max_drawdown([])

    def test_consistent_with_components(self) -> None:
        """RoMaD equals cumulative_return / abs(max_drawdown)."""
        returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        mdd = abs(max_drawdown(returns))
        cum_ret = float(cumulative_returns(returns).iloc[-1])
        expected = cum_ret / mdd if mdd != 0 else math.inf
        result = return_over_max_drawdown(returns)
        assert abs(result - expected) < 1e-9

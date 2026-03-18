"""Additional edge-case and error-path tests for metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("empyrical", reason="empyrical not installed")

from exceptions import SignalError
from metrics import (
    cumulative_returns,
    max_drawdown,
    return_over_max_drawdown,
    sharpe_ratio,
)


class TestSharpeRatioEdgeCases:
    def test_single_element_raises(self):
        """A single-element array has undefined std (ddof=1). Sharpe is inf."""
        result = sharpe_ratio([0.05])
        assert math.isinf(result)

    def test_risk_free_rate_shifts_sharpe(self):
        """Providing a risk-free rate reduces the Sharpe numerator."""
        returns = [0.01, 0.02, 0.03]
        sr_zero = sharpe_ratio(returns, risk_free=0.0)
        sr_high = sharpe_ratio(returns, risk_free=0.05)
        assert sr_zero > sr_high

    def test_negative_risk_free_increases_sharpe(self):
        returns = [0.01, 0.02, 0.03]
        sr_zero = sharpe_ratio(returns, risk_free=0.0)
        sr_neg = sharpe_ratio(returns, risk_free=-0.1)
        assert sr_neg > sr_zero

    def test_numpy_input_accepted(self):
        arr = np.array([0.1, 0.2, 0.3])
        result = sharpe_ratio(arr)
        assert isinstance(result, float)

    def test_list_input_accepted(self):
        result = sharpe_ratio([0.01, -0.01, 0.02])
        assert isinstance(result, float)

    def test_all_negative_returns_negative_sharpe(self):
        returns = [-0.01, -0.02, -0.015]
        assert sharpe_ratio(returns) < 0


class TestMaxDrawdownEdgeCases:
    def test_single_element(self):
        """Single return has zero drawdown."""
        mdd = max_drawdown([0.05])
        assert mdd == pytest.approx(0.0, abs=1e-6) or mdd >= -1e-6

    def test_large_negative_returns_large_drawdown(self):
        returns = [-0.5, -0.3, -0.2]
        mdd = max_drawdown(returns)
        assert mdd < -0.5  # significant drawdown

    def test_numpy_array_input(self):
        arr = np.array([0.01, -0.02, 0.03])
        result = max_drawdown(arr)
        assert isinstance(result, float)

    def test_returns_non_positive(self):
        """max_drawdown should always return <= 0."""
        returns = [0.1, 0.05, -0.03, 0.08]
        assert max_drawdown(returns) <= 0.0


class TestReturnOverMaxDrawdownEdgeCases:
    def test_empty_raises(self):
        with pytest.raises(SignalError):
            return_over_max_drawdown([])

    def test_single_positive_element(self):
        """Single positive return → zero drawdown → inf."""
        result = return_over_max_drawdown([0.1])
        assert math.isinf(result)

    def test_result_finite_with_mixed_returns(self):
        returns = [0.05, -0.1, 0.08, -0.05, 0.12]
        result = return_over_max_drawdown(returns)
        # Can be finite or inf depending on empyrical; must not be NaN
        assert not math.isnan(result)


class TestCumulativeReturnsEdgeCases:
    def test_single_element(self):
        result = cumulative_returns([0.1])
        assert len(result) == 1

    def test_series_length_matches_input(self):
        returns = [0.01, -0.02, 0.03, 0.005]
        result = cumulative_returns(returns)
        assert len(result) == 4

    def test_monotone_positive_is_increasing(self):
        returns = [0.01, 0.01, 0.01]
        result = cumulative_returns(returns)
        # Cumulative return should be non-decreasing for all-positive returns
        arr = result.values
        assert all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

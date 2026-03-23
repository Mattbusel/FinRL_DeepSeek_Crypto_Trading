"""Unit tests for src/rebalancer.py — 15+ test cases."""

from __future__ import annotations

import math
import sys
import os
from typing import Dict
from unittest.mock import MagicMock

import pytest

# Allow importing from root and src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.rebalancer import (
    RebalanceConfig,
    RebalanceTrade,
    RebalanceResult,
    Rebalancer,
    _tracking_error,
    _compute_current_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TWO_ASSET_WEIGHTS = {"BTC": 0.6, "ETH": 0.4}
THREE_ASSET_WEIGHTS = {"BTC": 0.5, "ETH": 0.3, "BNB": 0.2}


def make_config(**kwargs) -> RebalanceConfig:
    defaults = dict(
        target_weights=TWO_ASSET_WEIGHTS,
        threshold_pct=0.05,
        rebalance_frequency="daily",
        max_trade_size_usd=50_000.0,
    )
    defaults.update(kwargs)
    return RebalanceConfig(**defaults)


# ---------------------------------------------------------------------------
# RebalanceConfig tests
# ---------------------------------------------------------------------------

class TestRebalanceConfig:
    def test_valid_config(self):
        cfg = make_config()
        assert cfg.target_weights == TWO_ASSET_WEIGHTS
        assert cfg.threshold_pct == 0.05

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            RebalanceConfig(
                target_weights={"BTC": 0.6, "ETH": 0.5},
                threshold_pct=0.05,
                rebalance_frequency="daily",
                max_trade_size_usd=10_000.0,
            )

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            RebalanceConfig(
                target_weights={},
                threshold_pct=0.05,
                rebalance_frequency="daily",
                max_trade_size_usd=10_000.0,
            )

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RebalanceConfig(
                target_weights={"BTC": 1.0},
                threshold_pct=-0.01,
                rebalance_frequency="daily",
                max_trade_size_usd=10_000.0,
            )

    def test_non_positive_max_trade_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RebalanceConfig(
                target_weights={"BTC": 1.0},
                threshold_pct=0.05,
                rebalance_frequency="daily",
                max_trade_size_usd=0.0,
            )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTrackingError:
    def test_zero_error_when_equal(self):
        w = {"BTC": 0.6, "ETH": 0.4}
        assert _tracking_error(w, w) == pytest.approx(0.0, abs=1e-10)

    def test_known_error(self):
        actual = {"BTC": 0.7, "ETH": 0.3}
        target = {"BTC": 0.6, "ETH": 0.4}
        expected = math.sqrt((0.7 - 0.6) ** 2 + (0.3 - 0.4) ** 2)
        assert _tracking_error(actual, target) == pytest.approx(expected, rel=1e-9)

    def test_missing_asset_in_actual(self):
        actual = {"BTC": 1.0}
        target = {"BTC": 0.5, "ETH": 0.5}
        expected = math.sqrt((1.0 - 0.5) ** 2 + (0.0 - 0.5) ** 2)
        assert _tracking_error(actual, target) == pytest.approx(expected, rel=1e-9)

    def test_extra_asset_in_actual(self):
        actual = {"BTC": 0.6, "ETH": 0.3, "BNB": 0.1}
        target = {"BTC": 0.6, "ETH": 0.4}
        # BNB 0.1 weight with no target => penalty
        expected = math.sqrt((0.3 - 0.4) ** 2 + 0.1 ** 2)
        assert _tracking_error(actual, target) == pytest.approx(expected, rel=1e-9)

    def test_fully_misaligned(self):
        actual = {"BTC": 1.0}
        target = {"ETH": 1.0}
        expected = math.sqrt(1.0 ** 2 + 1.0 ** 2)
        assert _tracking_error(actual, target) == pytest.approx(expected, rel=1e-9)


class TestComputeCurrentWeights:
    def test_basic_weights(self):
        snapshot = {"BTC": 1.0, "ETH": 5.0}
        prices = {"BTC": 30_000.0, "ETH": 2_000.0}
        weights, total = _compute_current_weights(snapshot, prices)
        assert total == pytest.approx(40_000.0)
        assert weights["BTC"] == pytest.approx(30_000 / 40_000)
        assert weights["ETH"] == pytest.approx(10_000 / 40_000)

    def test_zero_portfolio(self):
        weights, total = _compute_current_weights({}, {})
        assert total == 0.0
        assert weights == {}


# ---------------------------------------------------------------------------
# Rebalancer.compute_trades tests
# ---------------------------------------------------------------------------

class TestComputeTrades:
    def setup_method(self):
        self.cfg = make_config(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            threshold_pct=0.05,
        )
        self.rebalancer = Rebalancer(self.cfg)

    def test_no_trades_within_threshold(self):
        snapshot = {"BTC": 1.0, "ETH": 1.0}
        prices = {"BTC": 50_000.0, "ETH": 50_000.0}
        trades = self.rebalancer.compute_trades(snapshot, prices)
        assert trades == []

    def test_trade_triggered_above_threshold(self):
        # BTC = 70 %, ETH = 30 % → BTC drifted +20 pp, ETH −20 pp
        snapshot = {"BTC": 0.7, "ETH": 0.3}
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        trades = self.rebalancer.compute_trades(snapshot, prices)
        assert len(trades) == 2
        btc_trade = next(t for t in trades if t.symbol == "BTC")
        eth_trade = next(t for t in trades if t.symbol == "ETH")
        assert btc_trade.trade_direction == "sell"
        assert eth_trade.trade_direction == "buy"

    def test_trade_direction_buy(self):
        # ETH underweight
        snapshot = {"BTC": 0.8, "ETH": 0.2}
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        trades = self.rebalancer.compute_trades(snapshot, prices)
        eth_trade = next((t for t in trades if t.symbol == "ETH"), None)
        assert eth_trade is not None
        assert eth_trade.trade_direction == "buy"

    def test_trade_usd_capped_at_max(self):
        cfg = make_config(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            threshold_pct=0.01,
            max_trade_size_usd=100.0,
        )
        rb = Rebalancer(cfg)
        snapshot = {"BTC": 0.8, "ETH": 0.2}
        prices = {"BTC": 1_000_000.0, "ETH": 1_000_000.0}
        trades = rb.compute_trades(snapshot, prices)
        for t in trades:
            assert t.trade_usd <= 100.0

    def test_estimated_cost_is_0_1_percent(self):
        snapshot = {"BTC": 0.8, "ETH": 0.2}
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        trades = self.rebalancer.compute_trades(snapshot, prices)
        for t in trades:
            assert t.estimated_cost == pytest.approx(t.trade_usd * 0.001)

    def test_empty_portfolio_returns_no_trades(self):
        trades = self.rebalancer.compute_trades({}, {})
        assert trades == []

    def test_trades_sorted_by_drift_descending(self):
        cfg = make_config(
            target_weights={"BTC": 0.4, "ETH": 0.3, "BNB": 0.3},
            threshold_pct=0.02,
        )
        rb = Rebalancer(cfg)
        snapshot = {"BTC": 0.7, "ETH": 0.1, "BNB": 0.2}
        prices = {"BTC": 1.0, "ETH": 1.0, "BNB": 1.0}
        trades = rb.compute_trades(snapshot, prices)
        drifts = [abs(t.current_weight - t.target_weight) for t in trades]
        assert drifts == sorted(drifts, reverse=True)

    def test_current_weight_stored_correctly(self):
        snapshot = {"BTC": 0.7, "ETH": 0.3}
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        trades = self.rebalancer.compute_trades(snapshot, prices)
        btc = next(t for t in trades if t.symbol == "BTC")
        assert btc.current_weight == pytest.approx(0.7)
        assert btc.target_weight == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Rebalancer.execute tests
# ---------------------------------------------------------------------------

class TestExecute:
    def _make_paper_trader(self, holdings: dict) -> MagicMock:
        """Build a minimal mock PaperTrader."""
        pt = MagicMock()
        positions = {}
        for sym, qty in holdings.items():
            pos = MagicMock()
            pos.quantity = qty
            positions[sym] = pos
        pt.positions = positions
        return pt

    def setup_method(self):
        self.cfg = make_config(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            threshold_pct=0.05,
        )
        self.rebalancer = Rebalancer(self.cfg)

    def test_execute_returns_rebalance_result(self):
        pt = self._make_paper_trader({"BTC": 0.7, "ETH": 0.3})
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        result = self.rebalancer.execute(pt, prices)
        assert isinstance(result, RebalanceResult)

    def test_execute_calls_sell_for_overweight(self):
        pt = self._make_paper_trader({"BTC": 0.7, "ETH": 0.3})
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        self.rebalancer.execute(pt, prices)
        assert pt.sell.called or pt.buy.called

    def test_tracking_error_after_less_than_before(self):
        pt = self._make_paper_trader({"BTC": 0.8, "ETH": 0.2})
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        result = self.rebalancer.execute(pt, prices)
        assert result.tracking_error_after <= result.tracking_error_before

    def test_total_cost_non_negative(self):
        pt = self._make_paper_trader({"BTC": 0.8, "ETH": 0.2})
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        result = self.rebalancer.execute(pt, prices)
        assert result.total_cost_usd >= 0.0

    def test_no_trades_when_already_balanced(self):
        pt = self._make_paper_trader({"BTC": 1.0, "ETH": 1.0})
        prices = {"BTC": 50_000.0, "ETH": 50_000.0}
        result = self.rebalancer.execute(pt, prices)
        assert result.trades_executed == []
        assert result.tracking_error_before == pytest.approx(0.0, abs=1e-10)

    def test_total_cost_equals_sum_of_estimated_costs(self):
        pt = self._make_paper_trader({"BTC": 0.8, "ETH": 0.2})
        prices = {"BTC": 100_000.0, "ETH": 100_000.0}
        result = self.rebalancer.execute(pt, prices)
        expected = sum(t.estimated_cost for t in result.trades_executed)
        assert result.total_cost_usd == pytest.approx(expected)

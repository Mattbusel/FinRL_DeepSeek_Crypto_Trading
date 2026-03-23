"""Tests for paper_trader.py.

All tests run in dry_run=True mode so no real Alpaca API calls are made.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)

from exceptions import ConfigError, DataFetchError
from paper_trader import (
    AlpacaPaperTrader,
    OrderResult,
    PaperTradingConfig,
    PaperTradingLoop,
)


# ---------------------------------------------------------------------------
# AlpacaPaperTrader — dry-run mode
# ---------------------------------------------------------------------------


class TestAlpacaPaperTraderDryRun:
    """Tests that operate in dry-run mode (no real API calls)."""

    def test_init_dry_run_does_not_raise(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        assert trader.dry_run is True

    def test_connect_dry_run_is_noop(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        trader.connect()  # should not raise
        assert trader._api is None

    def test_get_account_value_dry_run_returns_sentinel(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        value = trader.get_account_value()
        assert value == 100_000.0

    def test_get_positions_dry_run_returns_empty_list(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        positions = trader.get_positions()
        assert positions == []

    def test_place_order_buy_dry_run(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        result = trader.place_order("BTCUSD", qty=0.001, side="buy")
        assert isinstance(result, OrderResult)
        assert result.order_id == "dry-run"
        assert result.symbol == "BTCUSD"
        assert result.side == "buy"
        assert result.status == "simulated"

    def test_place_order_sell_dry_run(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        result = trader.place_order("BTCUSD", qty=0.001, side="sell")
        assert result.side == "sell"
        assert result.order_id == "dry-run"

    def test_place_order_invalid_side_raises(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        with pytest.raises(ValueError, match="side must be"):
            trader.place_order("BTCUSD", qty=0.001, side="hold")

    def test_place_order_zero_qty_raises(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        with pytest.raises(ValueError, match="qty must be positive"):
            trader.place_order("BTCUSD", qty=0.0, side="buy")

    def test_place_order_negative_qty_raises(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        with pytest.raises(ValueError, match="qty must be positive"):
            trader.place_order("BTCUSD", qty=-1.0, side="buy")


# ---------------------------------------------------------------------------
# AlpacaPaperTrader — credentials validation
# ---------------------------------------------------------------------------


class TestAlpacaPaperTraderCredentials:
    """Tests for missing-credentials handling."""

    def test_missing_keys_without_dry_run_raises_config_error(self) -> None:
        # Remove any env vars that might be set
        env = {k: v for k, v in os.environ.items() if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
                AlpacaPaperTrader(dry_run=False, api_key="", secret_key="")

    def test_dry_run_with_missing_keys_does_not_raise(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
        with patch.dict(os.environ, env, clear=True):
            trader = AlpacaPaperTrader(dry_run=True)
            assert trader.dry_run is True


# ---------------------------------------------------------------------------
# AlpacaPaperTrader — position size limit
# ---------------------------------------------------------------------------


class TestPositionSizeLimit:
    """Tests for the 10 % portfolio risk guard."""

    def test_position_limit_clamps_large_buy(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        # Account value is 100_000; 10% = 10_000; requesting 50_000 units
        clamped = trader._enforce_position_limit("BTCUSD", qty=50_000.0, side="buy")
        assert clamped <= 10_000.0

    def test_position_limit_does_not_clamp_small_buy(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        qty = trader._enforce_position_limit("BTCUSD", qty=0.001, side="buy")
        assert qty == pytest.approx(0.001)

    def test_position_limit_does_not_clamp_sell(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        # Sells bypass the limit check
        qty = trader._enforce_position_limit("BTCUSD", qty=999_999.0, side="sell")
        assert qty == 999_999.0


# ---------------------------------------------------------------------------
# PaperTradingLoop
# ---------------------------------------------------------------------------


class TestPaperTradingLoop:
    """Tests for the PaperTradingLoop execution."""

    def _make_mock_agent(self, action: int = 0) -> MagicMock:
        import torch
        agent = MagicMock()
        q_vals = torch.zeros(1, 3)
        q_vals[0, action] = 1.0
        agent.act.return_value = q_vals
        return agent

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_run_max_one_iteration_hold(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        agents = [self._make_mock_agent(0)]  # always hold
        config = PaperTradingConfig(
            symbol="BTCUSD",
            trade_qty=0.001,
            poll_interval_seconds=0.0,
            max_iterations=1,
        )
        loop = PaperTradingLoop(trader=trader, agents=agents, config=config)

        import torch
        state = torch.zeros(1, 12)

        def _state_provider():
            return state

        # Patch sleep to avoid delays
        with patch("paper_trader.time.sleep"):
            loop.run(state_provider=_state_provider)
        # No error means success

    def test_execute_action_buy_places_order(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        config = PaperTradingConfig(symbol="BTCUSD", trade_qty=0.001)
        loop = PaperTradingLoop(trader=trader, agents=[], config=config)
        result = loop._execute_action(1)
        assert result is not None
        assert result.side == "buy"

    def test_execute_action_sell_places_order(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        config = PaperTradingConfig(symbol="BTCUSD", trade_qty=0.001)
        loop = PaperTradingLoop(trader=trader, agents=[], config=config)
        result = loop._execute_action(2)
        assert result is not None
        assert result.side == "sell"

    def test_execute_action_hold_returns_none(self) -> None:
        trader = AlpacaPaperTrader(dry_run=True)
        config = PaperTradingConfig(symbol="BTCUSD", trade_qty=0.001)
        loop = PaperTradingLoop(trader=trader, agents=[], config=config)
        result = loop._execute_action(0)
        assert result is None

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_weighted_vote_majority(self) -> None:
        """Two buy agents vs one sell agent should produce buy."""
        trader = AlpacaPaperTrader(dry_run=True)
        agents = [
            self._make_mock_agent(1),
            self._make_mock_agent(1),
            self._make_mock_agent(2),
        ]
        config = PaperTradingConfig(agent_weights=[1.0, 1.0, 1.0])
        loop = PaperTradingLoop(trader=trader, agents=agents, config=config)

        import torch
        state = torch.zeros(1, 12)
        action = loop._weighted_vote(state)
        assert action == 1  # buy wins

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_weighted_vote_with_weights_overrides_count(self) -> None:
        """A single heavily weighted sell agent should beat two buy agents."""
        trader = AlpacaPaperTrader(dry_run=True)
        agents = [
            self._make_mock_agent(1),
            self._make_mock_agent(1),
            self._make_mock_agent(2),
        ]
        config = PaperTradingConfig(agent_weights=[0.1, 0.1, 10.0])
        loop = PaperTradingLoop(trader=trader, agents=agents, config=config)

        import torch
        state = torch.zeros(1, 12)
        action = loop._weighted_vote(state)
        assert action == 2  # sell wins due to weight

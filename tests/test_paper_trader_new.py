"""Unit tests for src/paper_trader.py — PaperTrader (Round 3)."""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.paper_trader import (
    PaperTrader,
    Position,
    PortfolioSnapshot,
    TradeResult,
    TradeStatus,
    TradeSide,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def trader() -> PaperTrader:
    return PaperTrader(initial_balance=10_000.0, commission_rate=0.001)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_initial_balance(trader: PaperTrader) -> None:
    assert trader.balance == 10_000.0


def test_initial_positions_empty(trader: PaperTrader) -> None:
    assert trader.positions == {}


def test_initial_history_empty(trader: PaperTrader) -> None:
    assert trader.history == []


def test_invalid_initial_balance_raises() -> None:
    with pytest.raises(ValueError):
        PaperTrader(initial_balance=-1.0)


def test_invalid_commission_raises() -> None:
    with pytest.raises(ValueError):
        PaperTrader(initial_balance=1_000.0, commission_rate=1.5)


# ---------------------------------------------------------------------------
# Buy
# ---------------------------------------------------------------------------

def test_buy_reduces_balance(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    # cost = 1000 + 1000*0.001 = 1001
    assert abs(trader.balance - 8_999.0) < 1.0


def test_buy_creates_position(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    assert "BTC" in trader.positions
    pos = trader.positions["BTC"]
    assert pos.side is TradeSide.LONG
    assert pos.quantity == pytest.approx(1.0)


def test_buy_filled_status(trader: PaperTrader) -> None:
    result = trader.buy("BTC", price=1_000.0, quantity=1.0)
    assert result.status is TradeStatus.FILLED


def test_buy_records_history(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    assert len(trader.history) == 1
    assert trader.history[0].side is TradeSide.LONG


def test_buy_insufficient_balance_rejected(trader: PaperTrader) -> None:
    result = trader.buy("BTC", price=50_000.0, quantity=1.0)
    assert result.status is TradeStatus.REJECTED
    # balance unchanged
    assert trader.balance == pytest.approx(10_000.0)


def test_buy_zero_price_rejected(trader: PaperTrader) -> None:
    result = trader.buy("BTC", price=0.0, quantity=1.0)
    assert result.status is TradeStatus.REJECTED


def test_buy_averages_into_existing_long(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    trader.buy("BTC", price=2_000.0, quantity=1.0)
    pos = trader.positions["BTC"]
    assert pos.quantity == pytest.approx(2.0)
    assert pos.entry_price == pytest.approx(1_500.0)


# ---------------------------------------------------------------------------
# Sell
# ---------------------------------------------------------------------------

def test_sell_long_removes_position(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    trader.sell("BTC", price=1_200.0, quantity=1.0)
    assert "BTC" not in trader.positions


def test_sell_long_increases_balance(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    balance_after_buy = trader.balance
    trader.sell("BTC", price=1_200.0, quantity=1.0)
    assert trader.balance > balance_after_buy


def test_sell_long_realized_pnl_positive(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    result = trader.sell("BTC", price=1_200.0, quantity=1.0)
    assert result.realized_pnl > 0.0


def test_sell_long_realized_pnl_negative(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    result = trader.sell("BTC", price=800.0, quantity=1.0)
    assert result.realized_pnl < 0.0


def test_sell_partial_position(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=2.0)
    trader.sell("BTC", price=1_200.0, quantity=1.0)
    assert "BTC" in trader.positions
    assert trader.positions["BTC"].quantity == pytest.approx(1.0)


def test_sell_opens_short_when_no_position(trader: PaperTrader) -> None:
    trader.sell("BTC", price=1_000.0, quantity=0.5)
    assert "BTC" in trader.positions
    assert trader.positions["BTC"].side is TradeSide.SHORT


def test_sell_zero_quantity_rejected(trader: PaperTrader) -> None:
    result = trader.sell("BTC", price=1_000.0, quantity=0.0)
    assert result.status is TradeStatus.REJECTED


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------

def test_mark_to_market_updates_unrealized_pnl(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    trader.mark_to_market({"BTC": 1_500.0})
    assert trader.positions["BTC"].unrealized_pnl == pytest.approx(500.0)


def test_mark_to_market_short_unrealized(trader: PaperTrader) -> None:
    trader.sell("BTC", price=1_000.0, quantity=1.0)
    trader.mark_to_market({"BTC": 800.0})
    assert trader.positions["BTC"].unrealized_pnl == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Close all
# ---------------------------------------------------------------------------

def test_close_all_clears_positions(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    trader.buy("ETH", price=100.0, quantity=5.0)
    trader.close_all({"BTC": 1_100.0, "ETH": 90.0})
    assert trader.positions == {}


def test_close_all_returns_trade_results(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    results = trader.close_all({"BTC": 1_100.0})
    assert len(results) == 1
    assert all(r.status is TradeStatus.FILLED for r in results)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def test_snapshot_returns_portfolio_snapshot(trader: PaperTrader) -> None:
    snap = trader.snapshot()
    assert isinstance(snap, PortfolioSnapshot)


def test_snapshot_total_equity_includes_positions(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    snap = trader.snapshot({"BTC": 1_500.0})
    assert snap.total_equity > trader.balance


def test_snapshot_total_pnl_correct_after_gain(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=1.0)
    snap = trader.snapshot({"BTC": 2_000.0})
    assert snap.total_pnl > 0


def test_snapshot_max_drawdown_non_negative(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=5.0)
    trader.mark_to_market({"BTC": 500.0})
    snap = trader.snapshot({"BTC": 500.0})
    assert snap.max_drawdown >= 0.0


# ---------------------------------------------------------------------------
# Drawdown tracking
# ---------------------------------------------------------------------------

def test_max_drawdown_increases_on_loss(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=5.0)
    trader.mark_to_market({"BTC": 500.0})
    assert trader.max_drawdown > 0.0


def test_max_drawdown_does_not_decrease(trader: PaperTrader) -> None:
    trader.buy("BTC", price=1_000.0, quantity=5.0)
    trader.mark_to_market({"BTC": 500.0})
    dd_after_loss = trader.max_drawdown
    trader.mark_to_market({"BTC": 2_000.0})
    assert trader.max_drawdown >= dd_after_loss

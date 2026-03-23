"""Unit tests for src/risk_limits.py — RiskLimits (Round 3)."""

from __future__ import annotations

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.paper_trader import PaperTrader, PortfolioSnapshot
from src.risk_limits import (
    DecisionKind,
    LimitConfig,
    RiskDecision,
    RiskLimits,
    RiskReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snap(
    balance: float = 10_000.0,
    total_equity: float = 10_000.0,
    daily_pnl: float = 0.0,
    total_pnl: float = 0.0,
    max_drawdown: float = 0.0,
    open_positions=None,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=time.time(),
        balance=balance,
        total_equity=total_equity,
        open_positions=open_positions or {},
        daily_pnl=daily_pnl,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
    )


@pytest.fixture()
def default_config() -> LimitConfig:
    return LimitConfig(
        max_position_pct=0.20,
        max_daily_loss_usd=500.0,
        max_drawdown_pct=0.10,
        max_leverage=3.0,
        position_size_usd=1_000.0,
    )


@pytest.fixture()
def limits(default_config: LimitConfig) -> RiskLimits:
    return RiskLimits(default_config)


# ---------------------------------------------------------------------------
# RiskDecision factories
# ---------------------------------------------------------------------------

def test_allow_is_allowed() -> None:
    d = RiskDecision.Allow()
    assert d.is_allowed is True
    assert d.kind is DecisionKind.ALLOW


def test_reject_is_not_allowed() -> None:
    d = RiskDecision.Reject("too risky")
    assert d.is_allowed is False
    assert d.kind is DecisionKind.REJECT
    assert d.reason == "too risky"


def test_reduce_is_allowed() -> None:
    d = RiskDecision.Reduce(0.5, "size down")
    assert d.is_allowed is True
    assert d.kind is DecisionKind.REDUCE
    assert d.scale_factor == pytest.approx(0.5)


def test_reduce_invalid_scale_raises() -> None:
    with pytest.raises(ValueError):
        RiskDecision.Reduce(0.0, "bad")


def test_reduce_scale_above_one_raises() -> None:
    with pytest.raises(ValueError):
        RiskDecision.Reduce(1.5, "bad")


# ---------------------------------------------------------------------------
# Daily loss checks
# ---------------------------------------------------------------------------

def test_daily_loss_within_limit_allowed(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-100.0)
    decision = limits.check("buy", snap)
    assert decision.is_allowed


def test_daily_loss_at_limit_rejected(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-500.0)
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT


def test_daily_loss_exceeded_rejected(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-1_000.0)
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT


def test_daily_gain_not_rejected(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=250.0)
    decision = limits.check("buy", snap)
    assert decision.is_allowed


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------

def test_drawdown_below_limit_allowed(limits: RiskLimits) -> None:
    snap = _make_snap(max_drawdown=0.05)
    decision = limits.check("buy", snap)
    assert decision.is_allowed


def test_drawdown_at_limit_rejected(limits: RiskLimits) -> None:
    snap = _make_snap(max_drawdown=0.10)
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT


def test_drawdown_exceeded_rejected(limits: RiskLimits) -> None:
    snap = _make_snap(max_drawdown=0.15)
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT


# ---------------------------------------------------------------------------
# Leverage checks
# ---------------------------------------------------------------------------

def test_zero_leverage_allowed(limits: RiskLimits) -> None:
    snap = _make_snap(balance=10_000.0)
    decision = limits.check("buy", snap)
    assert decision.is_allowed


def test_high_leverage_rejected(limits: RiskLimits) -> None:
    from src.paper_trader import Position, TradeSide
    pos = Position("BTC", TradeSide.LONG, entry_price=10_000.0, quantity=3.1)
    snap = _make_snap(
        balance=10_000.0,
        total_equity=41_000.0,
        open_positions={"BTC": pos},
    )
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT


# ---------------------------------------------------------------------------
# Concentration check
# ---------------------------------------------------------------------------

def test_concentration_within_limit_allowed(limits: RiskLimits) -> None:
    from src.paper_trader import Position, TradeSide
    # 10% of equity — well within 20% limit
    pos = Position("BTC", TradeSide.LONG, entry_price=1_000.0, quantity=1.0)
    snap = _make_snap(
        balance=9_000.0,
        total_equity=10_000.0,
        open_positions={"BTC": pos},
    )
    decision = limits.check("buy", snap)
    assert decision.is_allowed


def test_concentration_over_limit_reduces(limits: RiskLimits) -> None:
    from src.paper_trader import Position, TradeSide
    # 30% of equity — over 20% limit
    pos = Position("BTC", TradeSide.LONG, entry_price=3_000.0, quantity=1.0)
    snap = _make_snap(
        balance=7_000.0,
        total_equity=10_000.0,
        open_positions={"BTC": pos},
    )
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REDUCE
    assert decision.scale_factor < 1.0


# ---------------------------------------------------------------------------
# Check counting and report
# ---------------------------------------------------------------------------

def test_checks_run_increments(limits: RiskLimits) -> None:
    snap = _make_snap()
    limits.check("buy", snap)
    limits.check("sell", snap)
    report = limits.report()
    assert report.checks_run == 2


def test_rejections_counted(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-600.0)
    limits.check("buy", snap)
    report = limits.report()
    assert report.rejections == 1


def test_reductions_counted(limits: RiskLimits) -> None:
    from src.paper_trader import Position, TradeSide
    pos = Position("BTC", TradeSide.LONG, entry_price=3_000.0, quantity=1.0)
    snap = _make_snap(
        balance=7_000.0,
        total_equity=10_000.0,
        open_positions={"BTC": pos},
    )
    limits.check("buy", snap)
    report = limits.report()
    assert report.reductions == 1


def test_report_utilization_has_keys(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-100.0, max_drawdown=0.02)
    limits.check("buy", snap)
    report = limits.report()
    assert "daily_loss" in report.current_utilization
    assert "drawdown" in report.current_utilization


def test_report_utilization_values_in_range(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-250.0, max_drawdown=0.05)
    limits.check("buy", snap)
    report = limits.report()
    for val in report.current_utilization.values():
        assert 0.0 <= val <= 1.0


def test_reset_stats(limits: RiskLimits) -> None:
    snap = _make_snap(daily_pnl=-600.0)
    limits.check("buy", snap)
    limits.reset_stats()
    report = limits.report()
    assert report.checks_run == 0
    assert report.rejections == 0
    assert report.reductions == 0


# ---------------------------------------------------------------------------
# Integration: PaperTrader + RiskLimits
# ---------------------------------------------------------------------------

def test_integration_allows_normal_trade() -> None:
    trader = PaperTrader(initial_balance=10_000.0)
    limits = RiskLimits(LimitConfig())
    snap = trader.snapshot()
    decision = limits.check("buy", snap)
    assert decision.is_allowed
    if decision.is_allowed:
        result = trader.buy("BTC", price=500.0, quantity=1.0)
        assert result.status.value == "filled"


def test_integration_blocks_after_drawdown() -> None:
    trader = PaperTrader(initial_balance=10_000.0)
    limits = RiskLimits(LimitConfig(max_drawdown_pct=0.05))
    # Simulate large loss
    trader.buy("BTC", price=1_000.0, quantity=5.0)
    trader.mark_to_market({"BTC": 100.0})
    snap = trader.snapshot({"BTC": 100.0})
    decision = limits.check("buy", snap)
    assert decision.kind is DecisionKind.REJECT

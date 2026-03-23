"""Risk Limits Engine for LARSA.

Provides configurable hard stops and soft reductions that wrap any trade
action before it reaches the paper or live trading layer.

Example::

    from src.risk_limits import RiskLimits, LimitConfig
    from src.paper_trader import PaperTrader

    config = LimitConfig(
        max_position_pct=0.20,
        max_daily_loss_usd=500.0,
        max_drawdown_pct=0.10,
        max_leverage=3.0,
        position_size_usd=1_000.0,
    )
    limits = RiskLimits(config)
    trader = PaperTrader(10_000.0)
    snap = trader.snapshot()

    decision = limits.check("buy", snap)
    if decision.is_allowed:
        trader.buy("BTCUSDT", 30_000.0, 0.033)

    report = limits.report()
    print(report.checks_run, report.rejections)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from src.paper_trader import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LimitConfig:
    """Hard-stop and soft-reduction parameters.

    Attributes:
        max_position_pct: Maximum fraction of total equity in a single symbol
            (e.g. 0.20 = 20 %).
        max_daily_loss_usd: Maximum allowable loss within a single calendar day
            before all new trades are blocked (absolute USD / quote value).
        max_drawdown_pct: Maximum peak-to-trough drawdown fraction before the
            circuit breaker trips (e.g. 0.10 = 10 %).
        max_leverage: Maximum notional exposure as a multiple of cash balance.
        position_size_usd: Target nominal size of a single position; used by
            ``Reduce`` decisions to scale down oversized orders.
    """

    max_position_pct: float = 0.20
    max_daily_loss_usd: float = 500.0
    max_drawdown_pct: float = 0.10
    max_leverage: float = 3.0
    position_size_usd: float = 1_000.0


# ---------------------------------------------------------------------------
# RiskDecision
# ---------------------------------------------------------------------------

class DecisionKind(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE = "reduce"


@dataclass
class RiskDecision:
    """Outcome of a risk check.

    Attributes:
        kind: ``Allow``, ``Reject``, or ``Reduce``.
        reason: Human-readable explanation (empty string if allowed).
        scale_factor: Multiplier applied to order size on ``Reduce``
            decisions (1.0 for Allow/Reject).
    """

    kind: DecisionKind
    reason: str = ""
    scale_factor: float = 1.0

    @property
    def is_allowed(self) -> bool:
        return self.kind in (DecisionKind.ALLOW, DecisionKind.REDUCE)

    @classmethod
    def Allow(cls) -> "RiskDecision":  # noqa: N802
        return cls(kind=DecisionKind.ALLOW)

    @classmethod
    def Reject(cls, reason: str) -> "RiskDecision":  # noqa: N802
        return cls(kind=DecisionKind.REJECT, reason=reason, scale_factor=0.0)

    @classmethod
    def Reduce(cls, scale_factor: float, reason: str) -> "RiskDecision":  # noqa: N802
        if not (0.0 < scale_factor < 1.0):
            raise ValueError("scale_factor must be in (0, 1)")
        return cls(kind=DecisionKind.REDUCE, reason=reason, scale_factor=scale_factor)


# ---------------------------------------------------------------------------
# RiskReport
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """Aggregate statistics from the risk engine since last reset.

    Attributes:
        checks_run: Total number of ``check()`` calls.
        rejections: Number of ``Reject`` decisions issued.
        reductions: Number of ``Reduce`` decisions issued.
        current_utilization: Mapping of limit name -> current utilization
            fraction (0–1, where 1.0 = at the hard limit).
    """

    checks_run: int
    rejections: int
    reductions: int
    current_utilization: Dict[str, float]


# ---------------------------------------------------------------------------
# RiskLimits
# ---------------------------------------------------------------------------

class RiskLimits:
    """Evaluates proposed trade actions against configurable hard stops.

    Args:
        config: :class:`LimitConfig` with all hard-stop thresholds.
    """

    def __init__(self, config: LimitConfig) -> None:
        self._cfg = config
        self._checks_run = 0
        self._rejections = 0
        self._reductions = 0
        self._last_snapshot: Optional[PortfolioSnapshot] = None

    # ------------------------------------------------------------------
    # Internal checkers (each returns None if pass, or a RiskDecision)
    # ------------------------------------------------------------------

    def _check_daily_loss(self, snap: PortfolioSnapshot) -> Optional[RiskDecision]:
        if snap.daily_pnl < -abs(self._cfg.max_daily_loss_usd):
            return RiskDecision.Reject(
                f"daily loss limit breached: {snap.daily_pnl:.2f} USD "
                f"exceeds -{self._cfg.max_daily_loss_usd:.2f} USD"
            )
        return None

    def _check_drawdown(self, snap: PortfolioSnapshot) -> Optional[RiskDecision]:
        if snap.max_drawdown >= self._cfg.max_drawdown_pct:
            return RiskDecision.Reject(
                f"drawdown circuit breaker: {snap.max_drawdown * 100:.2f} % "
                f">= {self._cfg.max_drawdown_pct * 100:.2f} %"
            )
        return None

    def _check_leverage(self, snap: PortfolioSnapshot) -> Optional[RiskDecision]:
        if snap.balance <= 0:
            return RiskDecision.Reject("zero or negative cash balance")

        gross_notional = sum(
            pos.entry_price * pos.quantity
            for pos in snap.open_positions.values()
        )
        leverage = gross_notional / snap.balance if snap.balance > 0 else float("inf")

        if leverage >= self._cfg.max_leverage:
            return RiskDecision.Reject(
                f"leverage limit: current leverage {leverage:.2f}x "
                f">= max {self._cfg.max_leverage:.2f}x"
            )

        # Soft warning at 80 % of max leverage
        if leverage >= self._cfg.max_leverage * 0.80:
            scale = (self._cfg.max_leverage - leverage) / (self._cfg.max_leverage * 0.20)
            scale = max(0.1, min(0.9, scale))
            return RiskDecision.Reduce(
                scale_factor=scale,
                reason=(
                    f"approaching leverage limit ({leverage:.2f}x / "
                    f"{self._cfg.max_leverage:.2f}x); scaling to {scale:.0%}"
                ),
            )
        return None

    def _check_concentration(self, snap: PortfolioSnapshot) -> Optional[RiskDecision]:
        if snap.total_equity <= 0:
            return None

        for sym, pos in snap.open_positions.items():
            pos_value = pos.entry_price * pos.quantity
            pct = pos_value / snap.total_equity
            if pct > self._cfg.max_position_pct:
                return RiskDecision.Reduce(
                    scale_factor=self._cfg.max_position_pct / pct,
                    reason=(
                        f"{sym} position is {pct * 100:.1f} % of equity "
                        f"(limit {self._cfg.max_position_pct * 100:.0f} %)"
                    ),
                )
        return None

    def _check_position_size(self, snap: PortfolioSnapshot) -> Optional[RiskDecision]:
        """Soft-reduce if the balance would produce an oversized next position."""
        if snap.balance < self._cfg.position_size_usd:
            # Can still trade, but scale down to affordable size
            if snap.balance > 0:
                scale = snap.balance / self._cfg.position_size_usd
                if scale < 1.0:
                    return RiskDecision.Reduce(
                        scale_factor=max(0.1, scale),
                        reason=(
                            f"balance ({snap.balance:.2f}) below target "
                            f"position size ({self._cfg.position_size_usd:.2f})"
                        ),
                    )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        action: str,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> RiskDecision:
        """Evaluate all risk limits for a proposed *action*.

        Args:
            action: Intended trade action string, e.g. ``"buy"`` or ``"sell"``.
                Currently used for logging; all checks apply regardless of
                direction.
            portfolio_snapshot: Current portfolio state from
                :meth:`~src.paper_trader.PaperTrader.snapshot`.

        Returns:
            :class:`RiskDecision` — ``Allow``, ``Reject``, or ``Reduce``.
            The most restrictive applicable decision is returned; ``Reject``
            takes priority over ``Reduce``.
        """
        self._checks_run += 1
        self._last_snapshot = portfolio_snapshot

        checkers = [
            self._check_daily_loss,
            self._check_drawdown,
            self._check_leverage,
            self._check_concentration,
            self._check_position_size,
        ]

        reductions = []
        for checker in checkers:
            decision = checker(portfolio_snapshot)
            if decision is None:
                continue
            if decision.kind is DecisionKind.REJECT:
                self._rejections += 1
                return decision
            reductions.append(decision)

        if reductions:
            self._reductions += 1
            # Return the most conservative reduction (smallest scale_factor)
            return min(reductions, key=lambda d: d.scale_factor)

        return RiskDecision.Allow()

    def report(self) -> RiskReport:
        """Return aggregate statistics since instantiation.

        Returns:
            :class:`RiskReport` with check counts and current utilization.
        """
        snap = self._last_snapshot
        utilization: Dict[str, float] = {}

        if snap is not None:
            # Daily loss utilization
            if self._cfg.max_daily_loss_usd > 0:
                utilization["daily_loss"] = min(
                    1.0,
                    max(0.0, -snap.daily_pnl) / self._cfg.max_daily_loss_usd,
                )

            # Drawdown utilization
            if self._cfg.max_drawdown_pct > 0:
                utilization["drawdown"] = min(
                    1.0, snap.max_drawdown / self._cfg.max_drawdown_pct
                )

            # Leverage utilization
            gross_notional = sum(
                pos.entry_price * pos.quantity
                for pos in snap.open_positions.values()
            )
            if snap.balance > 0 and self._cfg.max_leverage > 0:
                utilization["leverage"] = min(
                    1.0,
                    (gross_notional / snap.balance) / self._cfg.max_leverage,
                )

            # Concentration utilization (worst single position)
            if snap.total_equity > 0 and self._cfg.max_position_pct > 0:
                worst_pct = max(
                    (pos.entry_price * pos.quantity / snap.total_equity
                     for pos in snap.open_positions.values()),
                    default=0.0,
                )
                utilization["concentration"] = min(
                    1.0, worst_pct / self._cfg.max_position_pct
                )

        return RiskReport(
            checks_run=self._checks_run,
            rejections=self._rejections,
            reductions=self._reductions,
            current_utilization=utilization,
        )

    def reset_stats(self) -> None:
        """Reset check/rejection/reduction counters (e.g. at session start)."""
        self._checks_run = 0
        self._rejections = 0
        self._reductions = 0

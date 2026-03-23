"""Drawdown-based risk control for portfolio management."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DrawdownStats:
    """Statistics describing the current drawdown state."""
    current_drawdown: float = 0.0
    max_drawdown_seen: float = 0.0
    peak_value: float = 0.0
    trough_value: float = 0.0
    days_in_drawdown: int = 0
    recovery_pct: float = 0.0


class DrawdownController:
    """Monitor and control portfolio exposure based on drawdown metrics.

    Parameters
    ----------
    max_drawdown:
        Hard limit; position sizing is zeroed at this level.
    soft_limit:
        Soft limit; linear scaling starts reducing at this level.
    recovery_factor:
        Fraction of the drawdown recovery required before restoring
        full position scale.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        soft_limit: float = 0.07,
        recovery_factor: float = 0.5,
    ) -> None:
        if soft_limit >= max_drawdown:
            raise ValueError(
                f"soft_limit ({soft_limit}) must be less than max_drawdown ({max_drawdown})"
            )
        self.max_drawdown = max_drawdown
        self.soft_limit = soft_limit
        self.recovery_factor = recovery_factor

        self._peak: float = 0.0
        self._trough: float = 0.0
        self._current_dd: float = 0.0
        self._max_dd_seen: float = 0.0
        self._days_in_dd: int = 0
        self._halted: bool = False

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, portfolio_value: float) -> dict:
        """Update drawdown state with the latest portfolio value.

        Returns a dict containing key metrics for logging.
        """
        if portfolio_value > self._peak:
            self._peak = portfolio_value
            if self._current_dd > 0.0:
                self._days_in_dd = 0  # reset counter on new peak
            self._current_dd = 0.0
            self._trough = portfolio_value
        else:
            dd = (self._peak - portfolio_value) / max(self._peak, 1e-12)
            self._current_dd = dd
            if portfolio_value < self._trough:
                self._trough = portfolio_value
            if dd > self._max_dd_seen:
                self._max_dd_seen = dd
            if dd > 0.0:
                self._days_in_dd += 1

        self._halted = self._current_dd >= self.max_drawdown

        return {
            "current_drawdown": self._current_dd,
            "max_drawdown_seen": self._max_dd_seen,
            "peak_value": self._peak,
            "trough_value": self._trough,
            "days_in_drawdown": self._days_in_dd,
            "halted": self._halted,
            "position_scale": self.position_scale(),
        }

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------

    def position_scale(self) -> float:
        """Return a multiplier in [0, 1] for position sizing.

        - 1.0  when drawdown < soft_limit
        - linear decay to 0.0 between soft_limit and max_drawdown
        - 0.0  when drawdown >= max_drawdown
        """
        if self._current_dd <= self.soft_limit:
            return 1.0
        if self._current_dd >= self.max_drawdown:
            return 0.0
        range_ = self.max_drawdown - self.soft_limit
        excess = self._current_dd - self.soft_limit
        return 1.0 - (excess / range_)

    def should_halt(self) -> bool:
        """True when the portfolio has reached the maximum drawdown limit."""
        return self._halted

    def recovery_progress(self) -> float:
        """Fraction of the recovery from the trough back to peak (0–1)."""
        if self._peak <= self._trough or self._peak <= 0.0:
            return 1.0  # No drawdown in progress.
        gap = self._peak - self._trough
        recovered = self._trough - self._trough  # Will be updated by update() calls.
        # Approximate from stored state.
        current_gap = self._peak * (1.0 - self._current_dd) - self._trough
        progress = current_gap / max(gap, 1e-12)
        return max(0.0, min(1.0, progress))

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------

    def trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        stop_pct: float,
    ) -> bool:
        """Return True if a long trailing stop has been hit.

        The stop fires when the price falls more than *stop_pct* below the
        entry price.
        """
        stop_level = entry_price * (1.0 - stop_pct)
        return current_price <= stop_level

    def max_position_size(
        self,
        capital: float,
        volatility: float,
        target_risk_pct: float = 0.02,
    ) -> float:
        """Compute volatility-adjusted maximum position size.

        Uses the fixed fractional formula:
            size = (capital * target_risk_pct) / volatility

        Scaled by the current position_scale() multiplier.
        """
        if volatility <= 0.0:
            return 0.0
        raw_size = (capital * target_risk_pct) / volatility
        return raw_size * self.position_scale()

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute half-Kelly fraction.

        Full Kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
        Half-Kelly  = full_kelly / 2

        Returns 0.0 if inputs are degenerate.
        """
        if avg_win <= 0.0 or avg_loss <= 0.0:
            return 0.0
        full_kelly = (win_rate / avg_loss) - ((1.0 - win_rate) / avg_win)
        half_kelly = full_kelly / 2.0
        return max(0.0, half_kelly)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> DrawdownStats:
        """Return a snapshot of current drawdown statistics."""
        return DrawdownStats(
            current_drawdown=self._current_dd,
            max_drawdown_seen=self._max_dd_seen,
            peak_value=self._peak,
            trough_value=self._trough,
            days_in_drawdown=self._days_in_dd,
            recovery_pct=self.recovery_progress() * 100.0,
        )

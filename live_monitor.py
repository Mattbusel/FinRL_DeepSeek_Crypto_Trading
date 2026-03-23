"""Live strategy monitoring dashboard (ASCII/terminal).

Renders an ASCII dashboard showing equity, P&L, drawdown, positions, and
signals for one or more active strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategyState:
    """Snapshot of a running strategy's current state."""

    strategy_name: str
    equity: float
    daily_pnl: float
    daily_pnl_pct: float          # e.g. 0.03 for +3 %
    drawdown: float                # e.g. 0.07 for 7 % drawdown (positive = down)
    positions: list[dict[str, Any]] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class LiveMonitor:
    """Maintains a history of strategy states and renders an ASCII dashboard."""

    _SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"

    def __init__(self) -> None:
        self._history: list[StrategyState] = []
        self._current: StrategyState | None = None

    def update(self, state: StrategyState) -> None:
        """Push a new state snapshot."""
        self._current = state
        self._history.append(state)
        # Keep at most 500 snapshots to bound memory.
        if len(self._history) > 500:
            self._history = self._history[-500:]

    # ── Rendering helpers ──────────────────────────────────────────────────

    @staticmethod
    def equity_sparkline(history: list[float], width: int = 40) -> str:
        """Render a Unicode sparkline for a list of equity values.

        Uses the 8 block characters ▁▂▃▄▅▆▇█.
        """
        if not history:
            return " " * width

        # Sample `width` points from history.
        n = len(history)
        sampled = [
            history[int(i * (n - 1) / (width - 1))] if width > 1 else history[-1]
            for i in range(width)
        ]

        lo, hi = min(sampled), max(sampled)
        span = hi - lo if hi != lo else 1.0

        chars = LiveMonitor._SPARKLINE_CHARS
        result = []
        for v in sampled:
            idx = int((v - lo) / span * (len(chars) - 1))
            idx = max(0, min(idx, len(chars) - 1))
            result.append(chars[idx])
        return "".join(result)

    @staticmethod
    def pnl_bar(pnl_pct: float, width: int = 20) -> str:
        """Render a +/- bar for a P&L percentage.

        Positive: ``+`` side filled with '█', remainder '░'.
        Negative: ``-`` side filled with '█', remainder '░'.
        """
        half = width // 2
        filled = int(min(abs(pnl_pct) * 100, 100) / 100 * half)
        filled = min(filled, half)
        empty = half - filled

        if pnl_pct >= 0:
            bar = "░" * half + "█" * filled + "░" * empty
        else:
            bar = "░" * empty + "█" * filled + "░" * half

        sign = "+" if pnl_pct >= 0 else "-"
        return f"[{bar}] {sign}{abs(pnl_pct) * 100:.2f}%"

    def alert_if_needed(self, state: StrategyState) -> list[str]:
        """Return warning messages for conditions that require attention."""
        warnings: list[str] = []

        if state.drawdown > 0.05:
            warnings.append(
                f"DRAWDOWN ALERT: {state.drawdown * 100:.1f}% drawdown exceeds 5% threshold"
            )

        # Position concentration: check if any single position > 50 % of equity.
        if state.equity > 0 and state.positions:
            for pos in state.positions:
                value = abs(float(pos.get("value", 0)))
                pct = value / state.equity
                if pct > 0.50:
                    ticker = pos.get("ticker", "UNKNOWN")
                    warnings.append(
                        f"CONCENTRATION ALERT: {ticker} is {pct * 100:.1f}% of equity"
                    )

        if state.daily_pnl_pct < -0.03:
            warnings.append(
                f"DAILY LOSS ALERT: {state.daily_pnl_pct * 100:.1f}% daily P&L"
            )

        return warnings

    # ── Full render ────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render the full ASCII dashboard for the current state."""
        if self._current is None:
            return "No data available.\n"

        state = self._current
        now_str = state.last_update.strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: list[str] = []

        # Header.
        title = f" LIVE MONITOR: {state.strategy_name} "
        border = "=" * max(60, len(title) + 4)
        lines.append(border)
        lines.append(title.center(len(border)))
        lines.append(f" Updated: {now_str}".ljust(len(border)))
        lines.append(border)

        # Equity row.
        lines.append(f"  Equity   : ${state.equity:>14,.2f}")

        # P&L bar.
        lines.append(f"  Daily P&L: ${state.daily_pnl:>+12,.2f}  {self.pnl_bar(state.daily_pnl_pct)}")

        # Drawdown meter.
        dd_pct = state.drawdown * 100
        dd_bar_width = 30
        dd_filled = int(min(dd_pct, 100) / 100 * dd_bar_width)
        dd_bar = "█" * dd_filled + "░" * (dd_bar_width - dd_filled)
        lines.append(f"  Drawdown : [{dd_bar}] {dd_pct:.2f}%")

        # Sparkline.
        equity_history = [s.equity for s in self._history]
        spark = self.equity_sparkline(equity_history)
        lines.append(f"  Equity   : {spark}")

        # Positions table.
        lines.append("")
        lines.append("  POSITIONS")
        lines.append("  " + "-" * 50)
        if state.positions:
            lines.append(f"  {'Ticker':<10} {'Size':>8} {'Value':>14} {'P&L':>10}")
            lines.append("  " + "-" * 50)
            for pos in state.positions:
                ticker = str(pos.get("ticker", "N/A"))[:10]
                size = pos.get("size", 0)
                value = pos.get("value", 0.0)
                pnl = pos.get("pnl", 0.0)
                lines.append(f"  {ticker:<10} {size:>8} {value:>14,.2f} {pnl:>+10,.2f}")
        else:
            lines.append("  (no open positions)")

        # Signals.
        lines.append("")
        lines.append("  RECENT SIGNALS")
        lines.append("  " + "-" * 50)
        if state.signals:
            for sig in state.signals[-5:]:  # last 5
                lines.append(f"  {sig}")
        else:
            lines.append("  (no recent signals)")

        # Alerts.
        alerts = self.alert_if_needed(state)
        if alerts:
            lines.append("")
            lines.append("  ALERTS")
            lines.append("  " + "-" * 50)
            for alert in alerts:
                lines.append(f"  *** {alert}")

        lines.append(border)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    monitor = LiveMonitor()
    state = StrategyState(
        strategy_name="DeepSeek-BTC",
        equity=100_000.0,
        daily_pnl=1_200.0,
        daily_pnl_pct=0.012,
        drawdown=0.02,
        positions=[{"ticker": "BTC", "size": 0.5, "value": 25000.0, "pnl": 500.0}],
        signals=["BUY BTC @ 50000", "SELL ETH @ 3200"],
    )
    monitor.update(state)
    print(monitor.render(), file=sys.stderr)
    print("LiveMonitor smoke test passed.", file=sys.stderr)

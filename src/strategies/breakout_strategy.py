"""Breakout detection and trading strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BreakoutSignal:
    symbol: str
    direction: str   # "up" or "down"
    price: float
    strength: float
    volume_confirm: bool


class BreakoutStrategy:
    """Detect and trade breakouts from a consolidation range."""

    def __init__(
        self,
        lookback: int = 20,
        breakout_threshold: float = 0.02,
        volume_factor: float = 1.5,
    ) -> None:
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.volume_factor = volume_factor

    # ------------------------------------------------------------------
    # Range computation
    # ------------------------------------------------------------------

    def compute_range(self, prices: List[float]) -> Tuple[float, float]:
        """Return (high, low) of the lookback window (excludes last element)."""
        window = prices[-self.lookback - 1 : -1] if len(prices) > 1 else prices
        if not window:
            return 0.0, 0.0
        return max(window), min(window)

    # ------------------------------------------------------------------
    # Volume confirmation
    # ------------------------------------------------------------------

    def is_volume_confirmed(self, volumes: List[float]) -> bool:
        """Return True if the latest volume > average of lookback window * volume_factor."""
        if len(volumes) < 2:
            return False
        window = volumes[-self.lookback - 1 : -1]
        if not window:
            return False
        avg_vol = sum(window) / len(window)
        return volumes[-1] > avg_vol * self.volume_factor

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def atr(self, prices: List[float]) -> float:
        """Average True Range over the lookback window (using close-to-close)."""
        window = prices[-self.lookback :] if len(prices) >= self.lookback else prices
        if len(window) < 2:
            return 0.0
        true_ranges = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
        return sum(true_ranges) / len(true_ranges)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(
        self,
        capital: float,
        price: float,
        atr_val: float,
        risk_pct: float = 0.01,
    ) -> int:
        """
        Size a position such that 1 ATR move equals risk_pct of capital.
        Returns 0 when ATR is zero to avoid division by zero.
        """
        if atr_val <= 0 or price <= 0:
            return 0
        risk_amount = capital * risk_pct
        shares = risk_amount / atr_val
        return max(0, int(shares))

    # ------------------------------------------------------------------
    # Breakout detection
    # ------------------------------------------------------------------

    def detect_breakout(
        self, prices: List[float], volumes: List[float]
    ) -> Optional[BreakoutSignal]:
        """
        Detect a breakout from the lookback range.
        Returns a BreakoutSignal or None when no breakout is detected.
        """
        if len(prices) <= self.lookback:
            return None

        high, low = self.compute_range(prices)
        current = prices[-1]
        vol_confirm = self.is_volume_confirmed(volumes)

        if high == 0:
            return None

        if current > high * (1.0 + self.breakout_threshold):
            strength = (current - high) / high
            return BreakoutSignal(
                symbol="",
                direction="up",
                price=current,
                strength=strength,
                volume_confirm=vol_confirm,
            )
        if low > 0 and current < low * (1.0 - self.breakout_threshold):
            strength = (low - current) / low
            return BreakoutSignal(
                symbol="",
                direction="down",
                price=current,
                strength=strength,
                volume_confirm=vol_confirm,
            )
        return None

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self, prices: List[float], volumes: List[float]
    ) -> dict:
        """
        Simple breakout backtest.
        Returns {trades, pnl, sharpe, win_rate}.
        """
        if len(prices) <= self.lookback + 1:
            return {"trades": [], "pnl": 0.0, "sharpe": 0.0, "win_rate": 0.0}

        trades = []
        position = 0.0
        entry_price = 0.0
        pnls: List[float] = []

        for i in range(self.lookback + 1, len(prices)):
            window_prices = prices[:i]
            window_volumes = volumes[:i] if i <= len(volumes) else volumes

            signal = self.detect_breakout(window_prices, window_volumes)

            if signal is not None and position == 0.0:
                # Enter
                position = 1.0 if signal.direction == "up" else -1.0
                entry_price = prices[i]
                trades.append({"entry": entry_price, "direction": signal.direction})

            elif position != 0.0:
                # Exit at next bar
                exit_price = prices[i]
                trade_pnl = position * (exit_price - entry_price)
                pnls.append(trade_pnl)
                if trades:
                    trades[-1]["exit"] = exit_price
                    trades[-1]["pnl"] = trade_pnl
                position = 0.0

        total_pnl = sum(pnls)
        win_rate = (
            sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
        )

        # Sharpe ratio (annualised, daily)
        if len(pnls) > 1:
            mean_pnl = total_pnl / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
            std_pnl = math.sqrt(variance) if variance > 0 else 1e-10
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
        else:
            sharpe = 0.0

        return {
            "trades": trades,
            "pnl": total_pnl,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }

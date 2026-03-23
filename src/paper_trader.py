"""Paper Trading Simulator for LARSA.

Tracks virtual positions, balance, and PnL in real-time without touching
real exchange APIs.  Designed for dry-run validation of live strategies.

Example::

    from src.paper_trader import PaperTrader

    trader = PaperTrader(initial_balance=10_000.0, commission_rate=0.001)
    result = trader.buy("BTCUSDT", price=30_000.0, quantity=0.1)
    print(result.status)               # TradeStatus.FILLED
    snap = trader.snapshot({"BTCUSDT": 31_000.0})
    print(snap.total_equity)           # ~13_097.0 after commission
    trader.mark_to_market({"BTCUSDT": 31_000.0})
    print(trader.positions["BTCUSDT"].unrealized_pnl)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """An open paper-trading position."""

    symbol: str
    side: TradeSide
    entry_price: float
    quantity: float
    entry_time: float = field(default_factory=time.time)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def mark(self, current_price: float) -> None:
        """Recompute unrealized PnL from *current_price*."""
        if self.side is TradeSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def market_value(self, current_price: float) -> float:
        """Current market value of this position in quote currency."""
        return current_price * self.quantity


# ---------------------------------------------------------------------------
# TradeResult
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    """Result of a single paper trade execution."""

    symbol: str
    side: TradeSide
    price: float
    quantity: float
    commission: float
    status: TradeStatus
    realized_pnl: float = 0.0
    timestamp: float = field(default_factory=time.time)
    message: str = ""


# ---------------------------------------------------------------------------
# PortfolioSnapshot
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    """Point-in-time state of the paper portfolio."""

    timestamp: float
    balance: float                          # free cash
    total_equity: float                     # cash + mark-to-market positions
    open_positions: Dict[str, Position]
    daily_pnl: float
    total_pnl: float
    max_drawdown: float                     # expressed as a positive fraction


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Virtual trading account that mirrors exchange mechanics without real money.

    Args:
        initial_balance: Starting cash balance in quote currency (e.g. USDT).
        commission_rate: Commission as a fraction of trade notional (default 0.1 %).
    """

    def __init__(
        self,
        initial_balance: float,
        commission_rate: float = 0.001,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not (0.0 <= commission_rate < 1.0):
            raise ValueError("commission_rate must be in [0, 1)")

        self._initial_balance = initial_balance
        self.balance: float = initial_balance
        self.commission_rate: float = commission_rate

        self.positions: Dict[str, Position] = {}
        self.history: List[TradeResult] = []

        # PnL / drawdown tracking
        self._peak_equity: float = initial_balance
        self._max_drawdown: float = 0.0
        self._day_start_equity: float = initial_balance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commission(self, notional: float) -> float:
        return notional * self.commission_rate

    def _total_equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        equity = self.balance
        for sym, pos in self.positions.items():
            price = (prices or {}).get(sym, pos.entry_price)
            equity += pos.market_value(price)
        return equity

    def _update_drawdown(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
    ) -> TradeResult:
        """Open or add to a long position.

        Returns a :class:`TradeResult` with status ``FILLED`` on success, or
        ``REJECTED`` if the account has insufficient balance.
        """
        if price <= 0 or quantity <= 0:
            return TradeResult(
                symbol=symbol,
                side=TradeSide.LONG,
                price=price,
                quantity=quantity,
                commission=0.0,
                status=TradeStatus.REJECTED,
                message="price and quantity must be positive",
            )

        notional = price * quantity
        commission = self._commission(notional)
        total_cost = notional + commission

        if total_cost > self.balance:
            return TradeResult(
                symbol=symbol,
                side=TradeSide.LONG,
                price=price,
                quantity=quantity,
                commission=commission,
                status=TradeStatus.REJECTED,
                message=f"insufficient balance ({self.balance:.2f} < {total_cost:.2f})",
            )

        self.balance -= total_cost

        if symbol in self.positions and self.positions[symbol].side is TradeSide.LONG:
            # Average into existing long
            existing = self.positions[symbol]
            total_qty = existing.quantity + quantity
            avg_price = (
                existing.entry_price * existing.quantity + price * quantity
            ) / total_qty
            existing.entry_price = avg_price
            existing.quantity = total_qty
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=TradeSide.LONG,
                entry_price=price,
                quantity=quantity,
            )

        result = TradeResult(
            symbol=symbol,
            side=TradeSide.LONG,
            price=price,
            quantity=quantity,
            commission=commission,
            status=TradeStatus.FILLED,
        )
        self.history.append(result)
        self._update_drawdown(self._total_equity())
        return result

    def sell(
        self,
        symbol: str,
        price: float,
        quantity: float,
    ) -> TradeResult:
        """Close or reduce a long position (or open/add to a short).

        If a long position exists for *symbol*, it will be reduced/closed and
        realized PnL computed.  If no position exists, a short position is
        opened (margin assumed unlimited for paper purposes).
        """
        if price <= 0 or quantity <= 0:
            return TradeResult(
                symbol=symbol,
                side=TradeSide.SHORT,
                price=price,
                quantity=quantity,
                commission=0.0,
                status=TradeStatus.REJECTED,
                message="price and quantity must be positive",
            )

        commission = self._commission(price * quantity)
        realized_pnl = 0.0

        if symbol in self.positions and self.positions[symbol].side is TradeSide.LONG:
            pos = self.positions[symbol]
            sell_qty = min(quantity, pos.quantity)
            realized_pnl = (price - pos.entry_price) * sell_qty - commission
            pos.realized_pnl += realized_pnl
            self.balance += price * sell_qty - commission

            if sell_qty >= pos.quantity:
                del self.positions[symbol]
            else:
                pos.quantity -= sell_qty
        else:
            # Open / add to short
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=TradeSide.SHORT,
                    entry_price=price,
                    quantity=quantity,
                )
            else:
                existing = self.positions[symbol]
                total_qty = existing.quantity + quantity
                avg_price = (
                    existing.entry_price * existing.quantity + price * quantity
                ) / total_qty
                existing.entry_price = avg_price
                existing.quantity = total_qty

            # Credit the short proceeds (minus commission) to balance
            self.balance += price * quantity - commission

        result = TradeResult(
            symbol=symbol,
            side=TradeSide.SHORT,
            price=price,
            quantity=quantity,
            commission=commission,
            status=TradeStatus.FILLED,
            realized_pnl=realized_pnl,
        )
        self.history.append(result)
        self._update_drawdown(self._total_equity())
        return result

    def close_all(self, prices: Dict[str, float]) -> List[TradeResult]:
        """Close every open position at the given prices.

        Args:
            prices: Mapping of symbol -> current price.

        Returns:
            List of :class:`TradeResult` for every position closed.
        """
        results: List[TradeResult] = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = prices.get(symbol, pos.entry_price)
            if pos.side is TradeSide.LONG:
                results.append(self.sell(symbol, price, pos.quantity))
            else:
                # Buy back short
                results.append(self.buy(symbol, price, pos.quantity))
        return results

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        """Update unrealized PnL for all open positions.

        Also refreshes the drawdown watermark with the latest equity value.

        Args:
            prices: Mapping of symbol -> current price.
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.mark(prices[symbol])
        equity = self._total_equity(prices)
        self._update_drawdown(equity)

    def snapshot(self, prices: Optional[Dict[str, float]] = None) -> PortfolioSnapshot:
        """Return a point-in-time snapshot of the portfolio.

        Args:
            prices: Optional mapping of symbol -> current price for MTM.

        Returns:
            :class:`PortfolioSnapshot` with current metrics.
        """
        if prices:
            self.mark_to_market(prices)

        equity = self._total_equity(prices or {})
        total_pnl = equity - self._initial_balance
        daily_pnl = equity - self._day_start_equity

        return PortfolioSnapshot(
            timestamp=time.time(),
            balance=self.balance,
            total_equity=equity,
            open_positions=dict(self.positions),
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            max_drawdown=self._max_drawdown,
        )

    def reset_daily(self) -> None:
        """Reset the daily PnL reference to current equity (call at day start)."""
        self._day_start_equity = self._total_equity()

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown observed since account inception (0–1 fraction)."""
        return self._max_drawdown

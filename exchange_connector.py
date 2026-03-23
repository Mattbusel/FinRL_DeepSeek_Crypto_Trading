"""Exchange API abstraction layer.

Provides an abstract ExchangeConnector interface and a MockExchangeConnector
for paper trading simulation with realistic slippage modeling.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Order Types ───────────────────────────────────────────────────────────────

class OrderType:
    """Order type constants."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


# ── Order Dataclass ───────────────────────────────────────────────────────────

@dataclass
class Order:
    """Represents a single order on an exchange."""
    order_id: str
    symbol: str
    side: str              # 'buy' | 'sell'
    order_type: str        # OrderType constant
    quantity: float
    price: Optional[float]  # None for market orders
    status: str            # 'pending' | 'filled' | 'cancelled' | 'partial'
    created_at: float = field(default_factory=time.time)
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0


# ── Abstract Base Class ───────────────────────────────────────────────────────

class ExchangeConnector(ABC):
    """Abstract exchange connector interface."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """Place an order on the exchange."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order. Returns True if successfully cancelled."""

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Return current balances as {currency: amount}."""

    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        """Return current position for a symbol."""

    @abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """Return the order book for a symbol."""


# ── Mock Exchange Connector ───────────────────────────────────────────────────

class MockExchangeConnector(ExchangeConnector):
    """A simulated exchange connector for backtesting and paper trading.

    - MARKET orders fill immediately at mid-price + slippage.
    - LIMIT orders fill if the simulated price crosses the limit.
    - Tracks balances, positions, and a synthetic order book.

    Args:
        initial_balance: Starting USDT balance.
        slippage_bps: Slippage in basis points applied to market fills.
        spread_bps: Bid-ask spread in basis points for order book generation.
    """

    DEFAULT_SLIPPAGE_BPS = 5   # 0.05%
    DEFAULT_SPREAD_BPS = 10    # 0.10%

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        spread_bps: int = DEFAULT_SPREAD_BPS,
    ) -> None:
        self._balance: Dict[str, float] = {"USDT": initial_balance}
        self._positions: Dict[str, float] = {}  # symbol → qty
        self._prices: Dict[str, float] = {}
        self._orders: Dict[str, Order] = {}
        self._slippage = slippage_bps / 10_000.0
        self._spread = spread_bps / 10_000.0

    # ── Price management ───────────────────────────────────────────────────

    def set_price(self, symbol: str, price: float) -> None:
        """Update the simulated mid-price for a symbol and try to fill pending limits."""
        self._prices[symbol] = price
        self._try_fill_limits(symbol)

    def _mid_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    # ── ExchangeConnector interface ────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="pending",
        )
        self._orders[order_id] = order

        if order_type == OrderType.MARKET:
            order = self.simulate_fill(order)
            self._orders[order_id] = order
        elif order_type == OrderType.LIMIT and price is not None:
            # Try to fill immediately if already crossed
            self._try_fill_limits(symbol)

        return self._orders[order_id]

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        order.status = "cancelled"
        return True

    def get_balance(self) -> Dict[str, float]:
        return dict(self._balance)

    def get_position(self, symbol: str) -> Dict:
        qty = self._positions.get(symbol, 0.0)
        mid = self._mid_price(symbol)
        return {
            "symbol": symbol,
            "quantity": qty,
            "market_value": qty * mid,
            "avg_entry_price": 0.0,  # simplified
        }

    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        mid = self._mid_price(symbol)
        half_spread = mid * self._spread / 2.0
        bids = [
            {"price": round(mid - half_spread - i * half_spread * 0.1, 4), "qty": 1.0}
            for i in range(depth)
        ]
        asks = [
            {"price": round(mid + half_spread + i * half_spread * 0.1, 4), "qty": 1.0}
            for i in range(depth)
        ]
        return {"symbol": symbol, "bids": bids, "asks": asks, "mid": mid}

    # ── Fill simulation ────────────────────────────────────────────────────

    def simulate_fill(self, order: Order) -> Order:
        """Apply slippage and fill an order at the simulated price.

        For MARKET orders: fills at mid-price ± slippage (buy pays more, sell receives less).
        For LIMIT orders: fills at the limit price if already crossed.
        """
        mid = self._mid_price(order.symbol)
        if mid <= 0.0:
            return order

        if order.order_type == OrderType.MARKET:
            if order.side == "buy":
                fill_price = mid * (1.0 + self._slippage)
            else:
                fill_price = mid * (1.0 - self._slippage)
        elif order.order_type == OrderType.LIMIT and order.price is not None:
            fill_price = order.price
        else:
            return order

        cost = fill_price * order.quantity

        if order.side == "buy":
            if self._balance.get("USDT", 0.0) < cost:
                order.status = "cancelled"
                return order
            self._balance["USDT"] = self._balance.get("USDT", 0.0) - cost
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0.0) + order.quantity
            )
        else:  # sell
            held = self._positions.get(order.symbol, 0.0)
            sell_qty = min(order.quantity, held)
            if sell_qty <= 0:
                order.status = "cancelled"
                return order
            self._positions[order.symbol] = held - sell_qty
            self._balance["USDT"] = self._balance.get("USDT", 0.0) + fill_price * sell_qty
            order.quantity = sell_qty  # adjust quantity to what was actually sold

        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.status = "filled"
        return order

    def _try_fill_limits(self, symbol: str) -> None:
        """Fill any pending limit orders that have been crossed."""
        mid = self._mid_price(symbol)
        for order_id, order in self._orders.items():
            if order.symbol != symbol or order.status != "pending":
                continue
            if order.order_type != OrderType.LIMIT or order.price is None:
                continue
            # Buy limit fills if mid ≤ limit price; sell limit fills if mid ≥ limit price
            if order.side == "buy" and mid <= order.price:
                self._orders[order_id] = self.simulate_fill(order)
            elif order.side == "sell" and mid >= order.price:
                self._orders[order_id] = self.simulate_fill(order)


# ── Paper Trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """High-level paper trading interface backed by MockExchangeConnector.

    Tracks realized and unrealized PnL per symbol.
    """

    def __init__(self, initial_balance: float = 100_000.0) -> None:
        self._exchange = MockExchangeConnector(initial_balance=initial_balance)
        self._trade_history: List[Order] = []
        # Track cost basis for PnL: symbol → (total_qty, total_cost)
        self._cost_basis: Dict[str, tuple] = {}
        self._realized_pnl: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Update the simulated price for a symbol."""
        self._exchange.set_price(symbol, price)

    def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: str = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Place a buy order."""
        order = self._exchange.place_order(
            symbol, "buy", order_type, quantity, price=limit_price
        )
        if order.status == "filled":
            qty, cost = self._cost_basis.get(symbol, (0.0, 0.0))
            self._cost_basis[symbol] = (
                qty + order.filled_qty,
                cost + order.avg_fill_price * order.filled_qty,
            )
        self._trade_history.append(order)
        return order

    def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: str = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Place a sell order."""
        order = self._exchange.place_order(
            symbol, "sell", order_type, quantity, price=limit_price
        )
        if order.status == "filled":
            qty, cost = self._cost_basis.get(symbol, (0.0, 0.0))
            if qty > 0:
                avg_cost = cost / qty
                realized = (order.avg_fill_price - avg_cost) * order.filled_qty
                self._realized_pnl[symbol] = (
                    self._realized_pnl.get(symbol, 0.0) + realized
                )
                new_qty = qty - order.filled_qty
                self._cost_basis[symbol] = (
                    max(new_qty, 0.0),
                    max(avg_cost * max(new_qty, 0.0), 0.0),
                )
        self._trade_history.append(order)
        return order

    def pnl_summary(self) -> Dict[str, Dict]:
        """Return realized and unrealized PnL per symbol."""
        summary = {}
        for symbol in set(list(self._realized_pnl.keys()) + list(self._cost_basis.keys())):
            position = self._exchange.get_position(symbol)
            qty, cost = self._cost_basis.get(symbol, (0.0, 0.0))
            mid = self._exchange._mid_price(symbol)
            unrealized = (mid - (cost / qty if qty > 0 else mid)) * position["quantity"]
            summary[symbol] = {
                "realized_pnl": self._realized_pnl.get(symbol, 0.0),
                "unrealized_pnl": unrealized,
                "position_qty": position["quantity"],
                "market_value": position["market_value"],
            }
        return summary

    def trade_history(self) -> List[Order]:
        """Return list of all placed orders."""
        return list(self._trade_history)

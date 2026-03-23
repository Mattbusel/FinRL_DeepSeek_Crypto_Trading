"""Unified exchange connector interface for crypto trading."""

from __future__ import annotations

import abc
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OrderBook:
    bids: List[List[float]]    # [[price, size], ...]
    asks: List[List[float]]    # [[price, size], ...]
    timestamp: float


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: float


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ExchangeConnector(abc.ABC):
    """Abstract base class for exchange connectors."""

    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Ticker:
        """Fetch the current ticker for a symbol."""
        ...

    @abc.abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        """Fetch the current order book."""
        ...

    @abc.abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> dict:
        """Place an order. Returns order dict with at least 'order_id' and 'status'."""
        ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True on success."""
        ...

    @abc.abstractmethod
    def get_balance(self) -> dict:
        """Return current account balance as {currency: amount}."""
        ...


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockExchangeConnector(ExchangeConnector):
    """Deterministic test double for ExchangeConnector.

    - Returns fixed mock data
    - Tracks placed and cancelled orders
    - Simulates partial fills (fills 80% of market orders)
    """

    PARTIAL_FILL_RATE = 0.80

    def __init__(self, initial_balance: Optional[Dict[str, float]] = None) -> None:
        self._balance: Dict[str, float] = initial_balance or {
            "USDT": 10_000.0,
            "BTC": 0.5,
            "ETH": 5.0,
        }
        self._orders: Dict[str, dict] = {}          # order_id -> order dict
        self._cancelled: set = set()                # set of cancelled order ids
        self._next_order_id = 1
        self._fill_history: List[dict] = []

    # ── Ticker ──────────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> Ticker:
        """Return deterministic mock ticker data."""
        prices = {
            "BTC/USDT": (49_900.0, 50_000.0, 49_950.0, 1234.5),
            "ETH/USDT": (2_990.0, 3_000.0, 2_995.0, 45_678.9),
            "SOL/USDT": (99.5, 100.5, 100.0, 987_654.3),
        }
        bid, ask, last, vol = prices.get(symbol, (99.0, 101.0, 100.0, 1000.0))
        return Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h=vol,
            timestamp=time.time(),
        )

    # ── Order book ──────────────────────────────────────────────────────────

    def get_orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        """Return deterministic mock order book."""
        mid = self.get_ticker(symbol).last
        spread = mid * 0.001  # 0.1% spread
        bids = [[mid - spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]
        asks = [[mid + spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]
        return OrderBook(bids=bids, asks=asks, timestamp=time.time())

    # ── Order management ────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> dict:
        """Place an order (deterministic mock). Simulates partial fills on MARKET orders."""
        order_id = f"MOCK-{self._next_order_id:06d}"
        self._next_order_id += 1

        ticker = self.get_ticker(symbol)
        exec_price = price if (order_type == "LIMIT" and price is not None) else (
            ticker.ask if side == "BUY" else ticker.bid
        )

        # Simulate partial fill for market orders
        filled_qty = quantity
        if order_type == "MARKET":
            filled_qty = quantity * self.PARTIAL_FILL_RATE

        status = "FILLED" if filled_qty >= quantity else "PARTIAL"

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "filled_quantity": filled_qty,
            "price": exec_price,
            "order_type": order_type,
            "status": status,
            "timestamp": time.time(),
        }
        self._orders[order_id] = order
        self._fill_history.append(order)

        # Update mock balance
        self._update_balance(side, symbol, filled_qty, exec_price)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if found and cancelled."""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELLED"
            self._cancelled.add(order_id)
            return True
        return False

    def get_balance(self) -> dict:
        """Return current mock balance."""
        return dict(self._balance)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _update_balance(
        self, side: str, symbol: str, quantity: float, price: float
    ) -> None:
        """Update mock balance after a fill."""
        base, quote = (symbol.split("/") + ["USDT"])[:2]
        notional = quantity * price
        if side == "BUY":
            self._balance[quote] = self._balance.get(quote, 0.0) - notional
            self._balance[base] = self._balance.get(base, 0.0) + quantity
        else:
            self._balance[base] = self._balance.get(base, 0.0) - quantity
            self._balance[quote] = self._balance.get(quote, 0.0) + notional

    def get_order(self, order_id: str) -> Optional[dict]:
        return self._orders.get(order_id)

    def list_orders(self) -> List[dict]:
        return list(self._orders.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMockExchangeConnector(unittest.TestCase):
    def setUp(self):
        self.conn = MockExchangeConnector()

    def test_get_ticker_returns_ticker(self):
        t = self.conn.get_ticker("BTC/USDT")
        self.assertIsInstance(t, Ticker)
        self.assertEqual(t.symbol, "BTC/USDT")

    def test_get_ticker_bid_ask_spread(self):
        t = self.conn.get_ticker("BTC/USDT")
        self.assertLess(t.bid, t.ask)

    def test_get_ticker_unknown_symbol(self):
        t = self.conn.get_ticker("UNKNOWN/USDT")
        self.assertIsInstance(t, Ticker)

    def test_get_orderbook_depth(self):
        ob = self.conn.get_orderbook("BTC/USDT", depth=5)
        self.assertEqual(len(ob.bids), 5)
        self.assertEqual(len(ob.asks), 5)

    def test_get_orderbook_bids_descending(self):
        ob = self.conn.get_orderbook("BTC/USDT", depth=5)
        bid_prices = [b[0] for b in ob.bids]
        self.assertEqual(bid_prices, sorted(bid_prices, reverse=True))

    def test_place_market_order_partial_fill(self):
        order = self.conn.place_order("BTC/USDT", "BUY", 1.0, "MARKET")
        self.assertAlmostEqual(order["filled_quantity"], 0.8, places=5)
        self.assertEqual(order["status"], "PARTIAL")

    def test_place_limit_order_full_fill(self):
        order = self.conn.place_order("BTC/USDT", "BUY", 0.1, "LIMIT", price=49000.0)
        self.assertAlmostEqual(order["filled_quantity"], 0.1, places=5)
        self.assertEqual(order["status"], "FILLED")

    def test_place_order_increments_id(self):
        o1 = self.conn.place_order("BTC/USDT", "BUY", 0.1, "LIMIT", price=49000.0)
        o2 = self.conn.place_order("BTC/USDT", "SELL", 0.1, "LIMIT", price=51000.0)
        self.assertNotEqual(o1["order_id"], o2["order_id"])

    def test_cancel_order_success(self):
        order = self.conn.place_order("ETH/USDT", "BUY", 1.0, "LIMIT", price=2900.0)
        result = self.conn.cancel_order(order["order_id"])
        self.assertTrue(result)
        self.assertEqual(self.conn.get_order(order["order_id"])["status"], "CANCELLED")

    def test_cancel_nonexistent_order(self):
        result = self.conn.cancel_order("NONEXISTENT")
        self.assertFalse(result)

    def test_get_balance_returns_dict(self):
        bal = self.conn.get_balance()
        self.assertIsInstance(bal, dict)
        self.assertIn("USDT", bal)

    def test_balance_updates_after_buy(self):
        initial_usdt = self.conn.get_balance()["USDT"]
        self.conn.place_order("BTC/USDT", "BUY", 0.1, "LIMIT", price=50000.0)
        new_usdt = self.conn.get_balance()["USDT"]
        self.assertLess(new_usdt, initial_usdt)

    def test_list_orders(self):
        self.conn.place_order("BTC/USDT", "BUY", 0.1, "LIMIT", price=49000.0)
        self.conn.place_order("ETH/USDT", "SELL", 1.0, "MARKET")
        orders = self.conn.list_orders()
        self.assertEqual(len(orders), 2)

    def test_connector_is_abstract(self):
        with self.assertRaises(TypeError):
            ExchangeConnector()  # type: ignore[abstract]

    def test_ticker_has_timestamp(self):
        t = self.conn.get_ticker("SOL/USDT")
        self.assertGreater(t.timestamp, 0)

    def test_orderbook_has_timestamp(self):
        ob = self.conn.get_orderbook("BTC/USDT")
        self.assertGreater(ob.timestamp, 0)


if __name__ == "__main__":
    unittest.main()

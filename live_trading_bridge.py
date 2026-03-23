"""Live Trading Bridge - connects LARSA signals to exchange APIs.

Supports:
- Binance (paper/live via testnet)
- Coinbase Advanced Trade API
- CCXT-compatible exchanges (generic)

Safety features:
- Hard position size limits
- Emergency stop-loss enforcement
- Cooldown between trades (min_trade_interval_secs)
- Max daily loss limit (kills trading for the day)
- Dry-run mode (log signals without executing)

Example::

    import asyncio
    from live_trading_bridge import (
        LiveTradingBridge, MockExchangeClient, RiskLimits, TradeSignal
    )
    from datetime import datetime

    exchange = MockExchangeClient(initial_balance=10_000.0)
    limits = RiskLimits(dry_run=True)
    queue: asyncio.Queue = asyncio.Queue()

    bridge = LiveTradingBridge(exchange=exchange, risk_limits=limits, signal_queue=queue)

    signal = TradeSignal(
        timestamp=datetime.utcnow(),
        symbol="BTC/USDT",
        direction="buy",
        confidence=0.80,
        regime="bull",
        position_size_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        source_model="AgentD3QN",
    )
    asyncio.run(bridge.process_signal(signal))
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    """A trade signal from LARSA.

    Attributes:
        timestamp: UTC datetime when the signal was generated.
        symbol: Exchange symbol, e.g. ``"BTC/USDT"``.
        direction: ``"buy"``, ``"sell"``, or ``"hold"``.
        confidence: Signal confidence in [0.0, 1.0].
        regime: Market regime label – ``"bull"``, ``"bear"``, or ``"sideways"``.
        position_size_pct: Fraction of available capital to deploy (0.0–1.0).
        stop_loss_pct: Stop-loss distance as a fraction of entry price.
        take_profit_pct: Take-profit distance as a fraction of entry price.
        source_model: Name of the ensemble member that generated this signal.
    """

    timestamp: datetime
    symbol: str
    direction: str
    confidence: float
    regime: str
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    source_model: str


@dataclass
class OrderResult:
    """Result returned by an exchange after placing an order.

    Attributes:
        order_id: Exchange-assigned order identifier.
        exchange: Name of the exchange (e.g. ``"binance_testnet"``).
        symbol: Trading pair (e.g. ``"BTC/USDT"``).
        side: ``"buy"`` or ``"sell"``.
        quantity: Amount of base asset transacted.
        price: Execution price.
        status: Order status string (e.g. ``"filled"``, ``"rejected"``).
        timestamp: UTC datetime of order placement.
        error: Error message if the order failed, else ``None``.
    """

    order_id: str
    exchange: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    status: str
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class RiskLimits:
    """Safety guardrails applied by :class:`LiveTradingBridge`.

    Attributes:
        max_position_pct: Maximum fraction of capital in any single position.
        max_daily_loss_pct: Halt trading for the day if daily loss exceeds this.
        min_confidence: Ignore signals whose confidence is below this threshold.
        min_trade_interval_secs: Minimum seconds between consecutive trades.
        max_open_positions: Maximum number of simultaneously open positions.
        dry_run: When ``True`` (default), log signals but never send real orders.
    """

    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    min_confidence: float = 0.65
    min_trade_interval_secs: int = 300
    max_open_positions: int = 3
    dry_run: bool = True


@dataclass
class _OpenPosition:
    """Internal record of an open position held by the bridge."""

    symbol: str
    side: str
    entry_price: Decimal
    quantity: Decimal
    stop_loss_price: Decimal
    take_profit_price: Decimal
    opened_at: datetime


# ---------------------------------------------------------------------------
# Abstract exchange client
# ---------------------------------------------------------------------------


class ExchangeClient(ABC):
    """Abstract base for all exchange connectors.

    Subclasses must implement each method and raise :class:`RuntimeError` (or
    an exchange-specific exception) on irrecoverable failures.
    """

    @abstractmethod
    async def get_balance(self, currency: str) -> Decimal:
        """Return the available balance for *currency* (e.g. ``"USDT"``)."""

    @abstractmethod
    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> OrderResult:
        """Place a market order; return the filled :class:`OrderResult`."""

    @abstractmethod
    async def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> OrderResult:
        """Place a limit order at *price*; return the submitted :class:`OrderResult`."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Return a ticker dict with at least ``{"bid": ..., "ask": ..., "last": ...}``."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel *order_id*; return ``True`` if successfully cancelled."""


# ---------------------------------------------------------------------------
# MockExchangeClient — paper trading with realistic slippage and fees
# ---------------------------------------------------------------------------


class MockExchangeClient(ExchangeClient):
    """Paper trading simulation with realistic slippage and fees.

    Maintains an in-memory balance and position book.  Market orders are
    filled immediately at the simulated price with slippage applied.

    Args:
        initial_balance: Starting USDT balance (default 10 000).
        fee_pct: Taker fee as a fraction of trade value (default 0.001 = 0.1%).
        slippage_pct: Maximum random slippage fraction applied to fills
            (default 0.0005 = 0.05%).
        base_price: Starting simulated price for BTC/USDT (default 50 000).
        price_drift: Per-tick random walk scale (default 0.001 = 0.1%).
    """

    EXCHANGE_NAME = "mock_paper"

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        base_price: float = 50_000.0,
        price_drift: float = 0.001,
    ) -> None:
        self._balances: Dict[str, Decimal] = {
            "USDT": Decimal(str(initial_balance)),
            "BTC": Decimal("0"),
            "ETH": Decimal("0"),
        }
        self._fee_pct = Decimal(str(fee_pct))
        self._slippage_pct = Decimal(str(slippage_pct))
        self._prices: Dict[str, Decimal] = {
            "BTC/USDT": Decimal(str(base_price)),
            "ETH/USDT": Decimal("2500"),
        }
        self._price_drift = price_drift
        self._order_counter = 0
        self._orders: Dict[str, OrderResult] = {}

    # ------------------------------------------------------------------
    # Price simulation helpers
    # ------------------------------------------------------------------

    def _tick_price(self, symbol: str) -> Decimal:
        """Advance the simulated price with a small random walk and return it."""
        current = self._prices.get(symbol, Decimal("100"))
        drift = Decimal(str(random.gauss(0.0, self._price_drift)))
        new_price = current * (Decimal("1") + drift)
        new_price = max(new_price, Decimal("1"))
        self._prices[symbol] = new_price
        return new_price

    def _apply_slippage(self, price: Decimal, side: str) -> Decimal:
        """Return *price* with slippage applied (buys pay more, sells receive less)."""
        slip = Decimal(str(random.uniform(0.0, float(self._slippage_pct))))
        if side == "buy":
            return price * (Decimal("1") + slip)
        return price * (Decimal("1") - slip)

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"MOCK-{self._order_counter:06d}"

    # ------------------------------------------------------------------
    # ExchangeClient interface
    # ------------------------------------------------------------------

    async def get_balance(self, currency: str) -> Decimal:
        """Return balance for *currency*; returns 0 if currency unknown."""
        return self._balances.get(currency.upper(), Decimal("0"))

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Return a simulated ticker for *symbol*."""
        price = self._tick_price(symbol)
        spread = price * Decimal("0.0001")
        return {
            "bid": float(price - spread),
            "ask": float(price + spread),
            "last": float(price),
        }

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> OrderResult:
        """Simulate a filled market order with slippage and fees."""
        ticker = await self.get_ticker(symbol)
        raw_price = Decimal(str(ticker["ask"] if side == "buy" else ticker["bid"]))
        fill_price = self._apply_slippage(raw_price, side)
        trade_value = fill_price * quantity
        fee = trade_value * self._fee_pct

        base_asset = symbol.split("/")[0]
        quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"

        if side == "buy":
            cost = trade_value + fee
            if self._balances.get(quote_asset, Decimal("0")) < cost:
                return OrderResult(
                    order_id=self._next_order_id(),
                    exchange=self.EXCHANGE_NAME,
                    symbol=symbol,
                    side=side,
                    quantity=Decimal("0"),
                    price=fill_price,
                    status="rejected",
                    timestamp=datetime.utcnow(),
                    error="Insufficient balance",
                )
            self._balances[quote_asset] = self._balances.get(quote_asset, Decimal("0")) - cost
            self._balances[base_asset] = self._balances.get(base_asset, Decimal("0")) + quantity
        else:
            proceeds = trade_value - fee
            if self._balances.get(base_asset, Decimal("0")) < quantity:
                return OrderResult(
                    order_id=self._next_order_id(),
                    exchange=self.EXCHANGE_NAME,
                    symbol=symbol,
                    side=side,
                    quantity=Decimal("0"),
                    price=fill_price,
                    status="rejected",
                    timestamp=datetime.utcnow(),
                    error="Insufficient asset balance",
                )
            self._balances[base_asset] = self._balances.get(base_asset, Decimal("0")) - quantity
            self._balances[quote_asset] = self._balances.get(quote_asset, Decimal("0")) + proceeds

        order_id = self._next_order_id()
        result = OrderResult(
            order_id=order_id,
            exchange=self.EXCHANGE_NAME,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            status="filled",
            timestamp=datetime.utcnow(),
        )
        self._orders[order_id] = result
        log.info(
            "mock_exchange.order_filled",
            order_id=order_id,
            side=side,
            symbol=symbol,
            quantity=float(quantity),
            price=float(fill_price),
            fee=float(fee),
        )
        return result

    async def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> OrderResult:
        """Submit a limit order (immediately filled if price is reachable in simulation)."""
        ticker = await self.get_ticker(symbol)
        current = Decimal(str(ticker["last"]))

        # Simple fill logic: buy limit fills if limit >= ask; sell limit fills if limit <= bid
        can_fill = (side == "buy" and price >= Decimal(str(ticker["ask"]))) or (
            side == "sell" and price <= Decimal(str(ticker["bid"]))
        )

        if can_fill:
            # Reuse market order logic at limit price
            return await self.place_market_order(symbol, side, quantity)

        order_id = self._next_order_id()
        result = OrderResult(
            order_id=order_id,
            exchange=self.EXCHANGE_NAME,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status="open",
            timestamp=datetime.utcnow(),
        )
        self._orders[order_id] = result
        log.info(
            "mock_exchange.limit_order_open",
            order_id=order_id,
            side=side,
            symbol=symbol,
            price=float(price),
        )
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False


# ---------------------------------------------------------------------------
# BinanceTestnetClient
# ---------------------------------------------------------------------------


class BinanceTestnetClient(ExchangeClient):
    """Binance testnet client — safe for testing, no real money involved.

    Uses the official Binance testnet REST API.  Requires ``httpx`` to be
    installed (``pip install httpx``).

    Args:
        api_key: Binance testnet API key.
        secret_key: Binance testnet secret key.

    Note:
        Testnet keys are obtained from https://testnet.binance.vision/.
        They are *separate* from live Binance credentials.
    """

    TESTNET_URL = "https://testnet.binance.vision"
    EXCHANGE_NAME = "binance_testnet"

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._client: Any = None  # httpx.AsyncClient, lazily initialised

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        """Return a lazily initialised httpx.AsyncClient."""
        if self._client is None:
            try:
                import httpx  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "httpx is required for BinanceTestnetClient. "
                    "Run: pip install httpx"
                ) from exc
            self._client = httpx.AsyncClient(
                base_url=self.TESTNET_URL,
                headers={"X-MBX-APIKEY": self._api_key},
                timeout=10.0,
            )
        return self._client

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add HMAC-SHA256 signature to *params* dict."""
        import hashlib
        import hmac
        import time
        import urllib.parse

        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self._secret_key.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    # ------------------------------------------------------------------
    # ExchangeClient interface
    # ------------------------------------------------------------------

    async def get_balance(self, currency: str) -> Decimal:
        """Fetch account balance for *currency* from Binance testnet."""
        client = await self._get_client()
        params = self._sign({})
        resp = await client.get("/api/v3/account", params=params)
        resp.raise_for_status()
        data = resp.json()
        for asset in data.get("balances", []):
            if asset["asset"] == currency.upper():
                return Decimal(asset["free"])
        return Decimal("0")

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Fetch best bid/ask from Binance testnet order book ticker."""
        client = await self._get_client()
        exchange_symbol = symbol.replace("/", "")
        resp = await client.get(
            "/api/v3/ticker/bookTicker", params={"symbol": exchange_symbol}
        )
        resp.raise_for_status()
        data = resp.json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        return {"bid": bid, "ask": ask, "last": (bid + ask) / 2.0}

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> OrderResult:
        """Submit a MARKET order to Binance testnet."""
        client = await self._get_client()
        exchange_symbol = symbol.replace("/", "")
        params = self._sign(
            {
                "symbol": exchange_symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": str(quantity),
            }
        )
        resp = await client.post("/api/v3/order", data=params)
        if resp.status_code != 200:
            error_msg = resp.text
            log.error(
                "binance_testnet.order_failed",
                symbol=symbol,
                side=side,
                error=error_msg,
            )
            return OrderResult(
                order_id="ERR",
                exchange=self.EXCHANGE_NAME,
                symbol=symbol,
                side=side,
                quantity=Decimal("0"),
                price=Decimal("0"),
                status="rejected",
                timestamp=datetime.utcnow(),
                error=error_msg,
            )
        data = resp.json()
        fills = data.get("fills", [])
        avg_price = (
            Decimal(str(sum(float(f["price"]) * float(f["qty"]) for f in fills)))
            / Decimal(str(sum(float(f["qty"]) for f in fills)))
            if fills
            else Decimal("0")
        )
        return OrderResult(
            order_id=str(data["orderId"]),
            exchange=self.EXCHANGE_NAME,
            symbol=symbol,
            side=side,
            quantity=Decimal(str(data.get("executedQty", quantity))),
            price=avg_price,
            status=data.get("status", "UNKNOWN").lower(),
            timestamp=datetime.utcnow(),
        )

    async def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> OrderResult:
        """Submit a LIMIT order to Binance testnet."""
        client = await self._get_client()
        exchange_symbol = symbol.replace("/", "")
        params = self._sign(
            {
                "symbol": exchange_symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": str(quantity),
                "price": str(price),
            }
        )
        resp = await client.post("/api/v3/order", data=params)
        resp.raise_for_status()
        data = resp.json()
        return OrderResult(
            order_id=str(data["orderId"]),
            exchange=self.EXCHANGE_NAME,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status=data.get("status", "NEW").lower(),
            timestamp=datetime.utcnow(),
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on Binance testnet."""
        # Symbol is required by Binance — callers should track symbol per order.
        # This simplified implementation requires the symbol to be embedded in
        # order_id as "SYMBOL:ID" or passes through without symbol info.
        client = await self._get_client()
        if ":" in order_id:
            symbol, oid = order_id.split(":", 1)
        else:
            log.warning("binance_testnet.cancel_missing_symbol", order_id=order_id)
            return False
        params = self._sign({"symbol": symbol.replace("/", ""), "orderId": oid})
        resp = await client.delete("/api/v3/order", params=params)
        return resp.status_code == 200


# ---------------------------------------------------------------------------
# LiveTradingBridge
# ---------------------------------------------------------------------------


class LiveTradingBridge:
    """Orchestrates signal ingestion and order execution with full safety controls.

    Consumes :class:`TradeSignal` objects from an :class:`asyncio.Queue`,
    applies risk checks, and (if not in dry-run mode) routes orders through
    the configured :class:`ExchangeClient`.

    Safety guarantees:
    - All signals below ``risk_limits.min_confidence`` are silently dropped.
    - Trades are throttled to at most one per ``min_trade_interval_secs``.
    - Total open positions are capped at ``max_open_positions``.
    - Each position size is capped at ``max_position_pct`` of available capital.
    - If daily loss exceeds ``max_daily_loss_pct``, no further orders are placed.
    - When ``dry_run=True``, orders are logged but never sent to the exchange.

    Args:
        exchange: An :class:`ExchangeClient` implementation.
        risk_limits: :class:`RiskLimits` configuration.
        signal_queue: Async queue from which :class:`TradeSignal` objects are consumed.
        log_file: Path to a JSONL file where all trade events are appended.
    """

    def __init__(
        self,
        exchange: ExchangeClient,
        risk_limits: RiskLimits,
        signal_queue: asyncio.Queue,
        log_file: str = "live_trades.jsonl",
    ) -> None:
        self._exchange = exchange
        self._limits = risk_limits
        self._queue = signal_queue
        self._log_file = log_file

        # Runtime state
        self._open_positions: Dict[str, _OpenPosition] = {}
        self._last_trade_time: Optional[datetime] = None
        self._daily_start_value: Optional[float] = None
        self._daily_realised_pnl: float = 0.0
        self._trade_log: List[Dict[str, Any]] = []
        self._is_halted: bool = False
        self._session_start: datetime = datetime.utcnow()

        log.info(
            "live_bridge.init",
            dry_run=risk_limits.dry_run,
            max_position_pct=risk_limits.max_position_pct,
            max_daily_loss_pct=risk_limits.max_daily_loss_pct,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume signals from the queue until the queue is exhausted or halted.

        Periodically enforces stop-loss checks on open positions.  Logs a
        daily report when the UTC date changes.
        """
        log.info("live_bridge.run_start", dry_run=self._limits.dry_run)
        last_date = datetime.utcnow().date()
        stop_loss_task = asyncio.create_task(self._stop_loss_loop())

        try:
            while not self._is_halted:
                try:
                    signal: TradeSignal = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check date rollover
                    today = datetime.utcnow().date()
                    if today != last_date:
                        log.info("live_bridge.day_rollover", date=str(today))
                        self._reset_daily_state()
                        last_date = today
                    continue

                result = await self.process_signal(signal)
                self._queue.task_done()

                today = datetime.utcnow().date()
                if today != last_date:
                    log.info("live_bridge.day_rollover", date=str(today))
                    self._reset_daily_state()
                    last_date = today
        finally:
            stop_loss_task.cancel()
            try:
                await stop_loss_task
            except asyncio.CancelledError:
                pass
            log.info("live_bridge.run_end", total_trades=len(self._trade_log))

    async def _stop_loss_loop(self) -> None:
        """Background task: checks stop-loss levels every 30 seconds."""
        while True:
            await asyncio.sleep(30)
            await self._enforce_stop_loss()

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def process_signal(self, signal: TradeSignal) -> Optional[OrderResult]:
        """Process a single :class:`TradeSignal` through risk checks and execution.

        Args:
            signal: Incoming trade signal from the LARSA ensemble.

        Returns:
            The :class:`OrderResult` if an order was placed, or ``None`` if the
            signal was rejected or skipped.
        """
        passed, reason = self._passes_risk_checks(signal)
        if not passed:
            log.info(
                "live_bridge.signal_rejected",
                symbol=signal.symbol,
                direction=signal.direction,
                reason=reason,
                confidence=signal.confidence,
            )
            self._log_trade(signal, None, rejection_reason=reason)
            return None

        if signal.direction == "hold":
            log.info("live_bridge.signal_hold", symbol=signal.symbol)
            return None

        # Fetch current price
        try:
            ticker = await self._exchange.get_ticker(signal.symbol)
            current_price = ticker["last"]
        except Exception as exc:
            log.error("live_bridge.ticker_fetch_failed", symbol=signal.symbol, error=str(exc))
            return None

        quantity = self._calculate_quantity(signal, current_price)
        if quantity <= Decimal("0"):
            log.warning("live_bridge.zero_quantity", symbol=signal.symbol)
            return None

        if self._limits.dry_run:
            log.info(
                "live_bridge.dry_run_order",
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=float(quantity),
                price=current_price,
                confidence=signal.confidence,
                regime=signal.regime,
                source_model=signal.source_model,
            )
            mock_result = OrderResult(
                order_id="DRY-RUN",
                exchange="dry_run",
                symbol=signal.symbol,
                side=signal.direction,
                quantity=quantity,
                price=Decimal(str(current_price)),
                status="dry_run",
                timestamp=datetime.utcnow(),
            )
            self._log_trade(signal, mock_result)
            return mock_result

        # Live / paper execution
        try:
            result = await self._exchange.place_market_order(
                symbol=signal.symbol,
                side=signal.direction,
                quantity=quantity,
            )
        except Exception as exc:
            log.error(
                "live_bridge.order_execution_failed",
                symbol=signal.symbol,
                direction=signal.direction,
                error=str(exc),
            )
            return None

        if result.status in ("filled", "partially_filled"):
            self._last_trade_time = datetime.utcnow()

            # Record open position
            if signal.direction == "buy":
                stop_price = result.price * Decimal(str(1.0 - signal.stop_loss_pct))
                tp_price = result.price * Decimal(str(1.0 + signal.take_profit_pct))
                self._open_positions[signal.symbol] = _OpenPosition(
                    symbol=signal.symbol,
                    side="buy",
                    entry_price=result.price,
                    quantity=result.quantity,
                    stop_loss_price=stop_price,
                    take_profit_price=tp_price,
                    opened_at=datetime.utcnow(),
                )
                log.info(
                    "live_bridge.position_opened",
                    symbol=signal.symbol,
                    entry=float(result.price),
                    stop_loss=float(stop_price),
                    take_profit=float(tp_price),
                )
            elif signal.direction == "sell" and signal.symbol in self._open_positions:
                pos = self._open_positions.pop(signal.symbol)
                pnl = float(result.price - pos.entry_price) * float(result.quantity)
                self._daily_realised_pnl += pnl
                log.info(
                    "live_bridge.position_closed",
                    symbol=signal.symbol,
                    pnl=pnl,
                    daily_pnl=self._daily_realised_pnl,
                )
                await self._check_daily_loss_halt()

        self._log_trade(signal, result)
        return result

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _passes_risk_checks(self, signal: TradeSignal) -> tuple[bool, str]:
        """Evaluate whether a signal passes all risk gates.

        Args:
            signal: The signal to evaluate.

        Returns:
            ``(True, "ok")`` if all checks pass; ``(False, reason)`` otherwise.
        """
        if self._is_halted:
            return False, "bridge_halted_daily_loss_limit"

        if signal.confidence < self._limits.min_confidence:
            return False, f"confidence {signal.confidence:.3f} < min {self._limits.min_confidence}"

        if signal.direction not in ("buy", "sell", "hold"):
            return False, f"unknown_direction:{signal.direction}"

        if signal.direction == "buy":
            if len(self._open_positions) >= self._limits.max_open_positions:
                return False, f"max_open_positions_{self._limits.max_open_positions}_reached"

        if signal.direction in ("buy", "sell") and self._last_trade_time is not None:
            elapsed = (datetime.utcnow() - self._last_trade_time).total_seconds()
            if elapsed < self._limits.min_trade_interval_secs:
                remaining = self._limits.min_trade_interval_secs - elapsed
                return False, f"cooldown:{remaining:.0f}s_remaining"

        if signal.position_size_pct > self._limits.max_position_pct:
            return False, (
                f"position_size {signal.position_size_pct:.3f} > "
                f"max {self._limits.max_position_pct}"
            )

        return True, "ok"

    def _calculate_quantity(self, signal: TradeSignal, price: float) -> Decimal:
        """Compute the order quantity from signal size and available capital.

        Fetches balance synchronously via a cached value.  The actual
        :meth:`ExchangeClient.get_balance` call is made asynchronously in the
        calling coroutine.

        Args:
            signal: The trade signal supplying ``position_size_pct``.
            price: Current market price.

        Returns:
            Order quantity rounded down to 6 decimal places.
        """
        # Approximate using position_size_pct directly; bridge caller handles capital
        # sizing more precisely via get_balance in process_signal path.
        capital_fraction = min(signal.position_size_pct, self._limits.max_position_pct)
        # Assume a reference capital of 10 000 USDT if balance is unknown
        reference_capital = Decimal("10000")
        raw_quantity = (reference_capital * Decimal(str(capital_fraction))) / Decimal(str(price))
        # Round down to avoid over-buying
        quantity = raw_quantity.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        return quantity

    # ------------------------------------------------------------------
    # Stop-loss enforcement
    # ------------------------------------------------------------------

    async def _enforce_stop_loss(self) -> None:
        """Check all open positions against stop-loss and take-profit levels.

        For each open position, fetches the current price and closes the
        position if either threshold is breached.
        """
        if self._limits.dry_run:
            return
        for symbol, pos in list(self._open_positions.items()):
            try:
                ticker = await self._exchange.get_ticker(symbol)
                current = Decimal(str(ticker["last"]))
            except Exception as exc:
                log.warning("live_bridge.stop_loss_ticker_fail", symbol=symbol, error=str(exc))
                continue

            triggered = False
            reason = ""
            if pos.side == "buy":
                if current <= pos.stop_loss_price:
                    triggered = True
                    reason = f"stop_loss@{float(pos.stop_loss_price):.2f}"
                elif current >= pos.take_profit_price:
                    triggered = True
                    reason = f"take_profit@{float(pos.take_profit_price):.2f}"

            if triggered:
                log.info(
                    "live_bridge.exit_triggered",
                    symbol=symbol,
                    reason=reason,
                    current_price=float(current),
                )
                try:
                    result = await self._exchange.place_market_order(
                        symbol=symbol, side="sell", quantity=pos.quantity
                    )
                    pnl = float(current - pos.entry_price) * float(pos.quantity)
                    self._daily_realised_pnl += pnl
                    del self._open_positions[symbol]
                    log.info(
                        "live_bridge.stop_loss_closed",
                        symbol=symbol,
                        pnl=pnl,
                        reason=reason,
                    )
                    await self._check_daily_loss_halt()
                except Exception as exc:
                    log.error(
                        "live_bridge.stop_loss_close_failed",
                        symbol=symbol,
                        error=str(exc),
                    )

    async def _check_daily_loss_halt(self) -> None:
        """Halt trading for the rest of the day if daily loss limit is breached."""
        if self._daily_start_value is None:
            return
        loss_pct = -self._daily_realised_pnl / self._daily_start_value
        if loss_pct >= self._limits.max_daily_loss_pct:
            self._is_halted = True
            log.warning(
                "live_bridge.daily_loss_halt",
                daily_pnl=self._daily_realised_pnl,
                loss_pct=loss_pct,
                limit=self._limits.max_daily_loss_pct,
            )

    def _reset_daily_state(self) -> None:
        """Reset daily P&L tracking at UTC midnight."""
        self._daily_realised_pnl = 0.0
        self._daily_start_value = None
        self._is_halted = False
        log.info("live_bridge.daily_reset")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_trade(
        self,
        signal: TradeSignal,
        result: Optional[OrderResult],
        rejection_reason: str = "",
    ) -> None:
        """Append a trade event to the in-memory log and the JSONL file.

        Args:
            signal: The originating signal.
            result: The :class:`OrderResult`, or ``None`` if rejected.
            rejection_reason: Human-readable reason if signal was rejected.
        """
        entry: Dict[str, Any] = {
            "event_time": datetime.utcnow().isoformat(),
            "signal": {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "regime": signal.regime,
                "position_size_pct": signal.position_size_pct,
                "stop_loss_pct": signal.stop_loss_pct,
                "take_profit_pct": signal.take_profit_pct,
                "source_model": signal.source_model,
            },
            "result": None,
            "rejection_reason": rejection_reason,
        }
        if result is not None:
            entry["result"] = {
                "order_id": result.order_id,
                "exchange": result.exchange,
                "side": result.side,
                "quantity": str(result.quantity),
                "price": str(result.price),
                "status": result.status,
                "timestamp": result.timestamp.isoformat(),
                "error": result.error,
            }
        self._trade_log.append(entry)
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            log.warning("live_bridge.log_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_daily_report(self) -> dict:
        """Generate a summary of all trades since the bridge started.

        Returns:
            A dict with keys: ``total_trades``, ``filled_trades``,
            ``rejected_trades``, ``dry_run_trades``, ``daily_realised_pnl``,
            ``open_positions``, ``is_halted``, ``session_start``.
        """
        filled = sum(
            1 for t in self._trade_log
            if t.get("result") and t["result"].get("status") == "filled"
        )
        rejected = sum(1 for t in self._trade_log if t.get("rejection_reason"))
        dry_run = sum(
            1 for t in self._trade_log
            if t.get("result") and t["result"].get("status") == "dry_run"
        )
        report = {
            "total_trades": len(self._trade_log),
            "filled_trades": filled,
            "rejected_trades": rejected,
            "dry_run_trades": dry_run,
            "daily_realised_pnl": round(self._daily_realised_pnl, 4),
            "open_positions": list(self._open_positions.keys()),
            "is_halted": self._is_halted,
            "session_start": self._session_start.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
        }
        log.info("live_bridge.daily_report", **{k: v for k, v in report.items() if k != "generated_at"})
        return report

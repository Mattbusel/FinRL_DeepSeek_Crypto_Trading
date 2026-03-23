"""Live Data Feed Adapter for crypto price streams.

Provides async WebSocket adapters for Binance and Coinbase exchange feeds,
a multiplexer for fan-out to multiple consumers, and a mock feed for testing.

Typical usage::

    from src.live_feed import BinanceFeed, FeedConfig, FeedMultiplexer

    config = FeedConfig(
        exchange="binance",
        symbols=["btcusdt", "ethusdt"],
        reconnect_delay=2.0,
        max_retries=5,
    )
    feed = BinanceFeed(config)

    async with feed.connect():
        async for tick in feed.stream():
            print(tick.symbol, tick.price)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeedConfig:
    """Configuration for a live exchange feed."""

    exchange: str
    symbols: List[str]
    reconnect_delay: float = 2.0
    max_retries: int = 10


@dataclass
class PriceTick:
    """A single price tick from an exchange."""

    symbol: str
    price: float
    volume: float
    timestamp: float  # Unix epoch seconds
    bid: float
    ask: float

    def spread(self) -> float:
        """Return bid-ask spread."""
        return self.ask - self.bid

    def mid(self) -> float:
        """Return mid-price."""
        return (self.bid + self.ask) / 2.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LiveFeed:
    """Abstract base for an async WebSocket price feed.

    Subclasses must implement:
    - ``_ws_url() -> str``
    - ``_subscribe_msg() -> object``  (JSON-serialisable subscription payload)
    - ``_parse(raw: str) -> Optional[PriceTick]``

    The ``connect()`` context manager handles reconnection.
    """

    def __init__(self, config: FeedConfig) -> None:
        self.config = config
        self._connected = False
        self._tick_queue: asyncio.Queue[Optional[PriceTick]] = asyncio.Queue()
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _ws_url(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def _subscribe_msg(self) -> object:  # pragma: no cover
        raise NotImplementedError

    def _parse(self, raw: str) -> Optional[PriceTick]:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> "_FeedContext":
        """Return an async context manager that manages the WS connection."""
        return _FeedContext(self)

    async def stream(self) -> AsyncGenerator[PriceTick, None]:
        """Yield PriceTick objects as they arrive."""
        while not self._stop_event.is_set():
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if tick is None:
                break
            yield tick

    async def _run(self) -> None:
        """Internal reconnect loop.  Runs until stop() is called."""
        retries = 0
        while not self._stop_event.is_set():
            try:
                await self._ws_session()
                retries = 0
            except Exception as exc:
                retries += 1
                if self.config.max_retries >= 0 and retries > self.config.max_retries:
                    log.error(
                        "Max retries (%d) exceeded: %s", self.config.max_retries, exc
                    )
                    break
                delay = self.config.reconnect_delay * min(retries, 10)
                log.warning(
                    "Feed disconnected (attempt %d/%d): %s — retrying in %.1fs",
                    retries,
                    self.config.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        await self._tick_queue.put(None)  # sentinel

    async def _ws_session(self) -> None:
        """Single WebSocket session.  Subclasses may override for mocking."""
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "websockets package is required: pip install websockets"
            ) from exc

        url = self._ws_url()
        log.info("Connecting to %s", url)
        async with websockets.connect(url) as ws:
            sub = self._subscribe_msg()
            if sub is not None:
                await ws.send(json.dumps(sub))
            self._connected = True
            async for message in ws:
                if self._stop_event.is_set():
                    break
                tick = self._parse(message)
                if tick is not None:
                    await self._tick_queue.put(tick)

    def stop(self) -> None:
        """Signal the feed to stop reconnecting."""
        self._stop_event.set()


class _FeedContext:
    """Async context manager returned by LiveFeed.connect()."""

    def __init__(self, feed: LiveFeed) -> None:
        self._feed = feed
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    async def __aenter__(self) -> LiveFeed:
        self._task = asyncio.create_task(self._feed._run())
        return self._feed

    async def __aexit__(self, *_: object) -> None:
        self._feed.stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()


# ---------------------------------------------------------------------------
# Binance Feed
# ---------------------------------------------------------------------------


class BinanceFeed(LiveFeed):
    """WebSocket feed for Binance combined stream.

    URL pattern:
        wss://stream.binance.com:9443/stream?streams=<sym1>@bookTicker/<sym2>@bookTicker
    """

    _BASE = "wss://stream.binance.com:9443"

    def _ws_url(self) -> str:
        streams = "/".join(
            f"{s.lower()}@bookTicker" for s in self.config.symbols
        )
        return f"{self._BASE}/stream?streams={streams}"

    def _subscribe_msg(self) -> object:
        # Combined stream: no explicit subscription needed; topics encoded in URL.
        return None

    def _parse(self, raw: str) -> Optional[PriceTick]:
        """Parse Binance combined stream bookTicker message.

        Expected format::

            {
              "stream": "btcusdt@bookTicker",
              "data": {
                "u": 123,        # update id
                "s": "BTCUSDT",  # symbol
                "b": "30000.00", # best bid price
                "B": "1.5",      # best bid qty
                "a": "30001.00", # best ask price
                "A": "0.8"       # best ask qty
              }
            }
        """
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
            symbol: str = data["s"]
            bid = float(data["b"])
            ask = float(data["a"])
            bid_qty = float(data.get("B", 0.0))
            ask_qty = float(data.get("A", 0.0))
            price = (bid + ask) / 2.0
            volume = (bid_qty + ask_qty) / 2.0
            return PriceTick(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=time.time(),
                bid=bid,
                ask=ask,
            )
        except (KeyError, ValueError, TypeError) as exc:
            log.debug("Failed to parse Binance message: %s — %s", raw[:200], exc)
            return None


# ---------------------------------------------------------------------------
# Coinbase Feed
# ---------------------------------------------------------------------------


class CoinbaseFeed(LiveFeed):
    """WebSocket feed for Coinbase Advanced Trade API.

    URL: wss://advanced-trade-ws.coinbase.com
    """

    _URL = "wss://advanced-trade-ws.coinbase.com"

    def _ws_url(self) -> str:
        return self._URL

    def _subscribe_msg(self) -> object:
        return {
            "type": "subscribe",
            "product_ids": [s.upper() for s in self.config.symbols],
            "channel": "ticker",
        }

    def _parse(self, raw: str) -> Optional[PriceTick]:
        """Parse Coinbase Advanced Trade ticker channel message.

        Expected format::

            {
              "channel": "ticker",
              "events": [
                {
                  "type": "update",
                  "tickers": [
                    {
                      "product_id": "BTC-USD",
                      "price": "30000.00",
                      "volume_24_h": "1234.56",
                      "best_bid": "29999.00",
                      "best_ask": "30001.00"
                    }
                  ]
                }
              ]
            }
        """
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "ticker":
                return None
            events = msg.get("events", [])
            ticks = []
            for event in events:
                for t in event.get("tickers", []):
                    bid = float(t.get("best_bid", 0.0) or 0.0)
                    ask = float(t.get("best_ask", 0.0) or 0.0)
                    price = float(t.get("price", 0.0) or 0.0)
                    if price == 0.0:
                        price = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
                    ticks.append(
                        PriceTick(
                            symbol=t["product_id"],
                            price=price,
                            volume=float(t.get("volume_24_h", 0.0) or 0.0),
                            timestamp=time.time(),
                            bid=bid,
                            ask=ask,
                        )
                    )
            return ticks[0] if ticks else None
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            log.debug("Failed to parse Coinbase message: %s — %s", raw[:200], exc)
            return None


# ---------------------------------------------------------------------------
# Mock Feed (for testing)
# ---------------------------------------------------------------------------


class MockFeed(LiveFeed):
    """Deterministic feed that yields a predetermined list of PriceTicks.

    Useful for unit testing without a network connection.

    Example::

        ticks = [PriceTick("BTC", 30000.0, 1.0, time.time(), 29999.0, 30001.0)]
        feed = MockFeed(ticks)
        async with feed.connect():
            results = []
            async for tick in feed.stream():
                results.append(tick)
        assert results == ticks
    """

    def __init__(self, ticks: List[PriceTick], delay: float = 0.0) -> None:
        config = FeedConfig(exchange="mock", symbols=[], reconnect_delay=0.0, max_retries=0)
        super().__init__(config)
        self._ticks = ticks
        self._delay = delay

    def _ws_url(self) -> str:
        return "mock://"

    def _subscribe_msg(self) -> object:
        return None

    def _parse(self, raw: str) -> Optional[PriceTick]:
        return None

    async def _ws_session(self) -> None:
        for tick in self._ticks:
            if self._stop_event.is_set():
                break
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            await self._tick_queue.put(tick)
        # Session ends after all ticks; reconnect loop will stop because
        # max_retries=0 means one attempt.

    async def _run(self) -> None:
        await self._ws_session()
        await self._tick_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Feed Multiplexer
# ---------------------------------------------------------------------------


class FeedMultiplexer:
    """Fan out ticks from a single LiveFeed to multiple asyncio.Queue consumers.

    Example::

        mux = FeedMultiplexer(feed)
        q1 = mux.add_consumer()
        q2 = mux.add_consumer()

        async with feed.connect():
            await mux.run()   # pump ticks to all queues

        # q1 and q2 both received all ticks.
    """

    def __init__(self, feed: LiveFeed, maxsize: int = 0) -> None:
        self._feed = feed
        self._maxsize = maxsize
        self._consumers: List[asyncio.Queue[Optional[PriceTick]]] = []

    def add_consumer(self) -> "asyncio.Queue[Optional[PriceTick]]":
        """Register a new consumer queue and return it."""
        q: asyncio.Queue[Optional[PriceTick]] = asyncio.Queue(maxsize=self._maxsize)
        self._consumers.append(q)
        return q

    def consumer_count(self) -> int:
        return len(self._consumers)

    async def run(self) -> None:
        """Pump ticks from the feed to all registered consumer queues.

        Returns when the feed is exhausted (None sentinel received).
        """
        async for tick in self._feed.stream():
            for q in self._consumers:
                try:
                    q.put_nowait(tick)
                except asyncio.QueueFull:
                    log.warning(
                        "Consumer queue full (%d items) — dropping tick %s",
                        q.maxsize,
                        tick.symbol,
                    )
        # Broadcast sentinel to all consumers
        for q in self._consumers:
            await q.put(None)

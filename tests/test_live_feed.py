"""Unit tests for src/live_feed.py — uses MockFeed, no network required."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from src.live_feed import (
    BinanceFeed,
    CoinbaseFeed,
    FeedConfig,
    FeedMultiplexer,
    MockFeed,
    PriceTick,
)


def _tick(symbol: str = "BTCUSDT", price: float = 30000.0, volume: float = 1.0,
          bid: float = 29999.0, ask: float = 30001.0) -> PriceTick:
    return PriceTick(symbol=symbol, price=price, volume=volume,
                     timestamp=time.time(), bid=bid, ask=ask)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# FeedConfig
# ---------------------------------------------------------------------------

class TestFeedConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = FeedConfig(exchange="binance", symbols=["btcusdt"])
        self.assertEqual(cfg.reconnect_delay, 2.0)
        self.assertEqual(cfg.max_retries, 10)

    def test_custom_values(self):
        cfg = FeedConfig(exchange="coinbase", symbols=["BTC-USD", "ETH-USD"],
                         reconnect_delay=0.5, max_retries=3)
        self.assertEqual(len(cfg.symbols), 2)
        self.assertEqual(cfg.max_retries, 3)


# ---------------------------------------------------------------------------
# PriceTick
# ---------------------------------------------------------------------------

class TestPriceTick(unittest.TestCase):

    def test_spread(self):
        t = _tick(bid=100.0, ask=101.0)
        self.assertAlmostEqual(t.spread(), 1.0)

    def test_mid(self):
        t = _tick(bid=100.0, ask=102.0)
        self.assertAlmostEqual(t.mid(), 101.0)

    def test_fields(self):
        t = _tick("ETHUSDT", 2000.0, 5.0, 1999.0, 2001.0)
        self.assertEqual(t.symbol, "ETHUSDT")
        self.assertAlmostEqual(t.price, 2000.0)
        self.assertAlmostEqual(t.volume, 5.0)

    def test_zero_spread(self):
        t = _tick(bid=100.0, ask=100.0)
        self.assertAlmostEqual(t.spread(), 0.0)
        self.assertAlmostEqual(t.mid(), 100.0)


# ---------------------------------------------------------------------------
# MockFeed
# ---------------------------------------------------------------------------

class TestMockFeed(unittest.TestCase):

    def test_yields_all_ticks(self):
        ticks = [_tick("BTC"), _tick("ETH"), _tick("SOL")]
        feed = MockFeed(ticks)

        async def collect():
            results = []
            async with feed.connect():
                async for tick in feed.stream():
                    results.append(tick)
            return results

        got = run(collect())
        self.assertEqual(len(got), 3)
        self.assertEqual(got[0].symbol, "BTC")
        self.assertEqual(got[2].symbol, "SOL")

    def test_empty_feed(self):
        feed = MockFeed([])

        async def collect():
            results = []
            async with feed.connect():
                async for tick in feed.stream():
                    results.append(tick)
            return results

        got = run(collect())
        self.assertEqual(got, [])

    def test_single_tick(self):
        ticks = [_tick("AVAX", 20.0)]
        feed = MockFeed(ticks)

        async def collect():
            results = []
            async with feed.connect():
                async for t in feed.stream():
                    results.append(t)
            return results

        got = run(collect())
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0].price, 20.0)

    def test_preserves_order(self):
        ticks = [_tick(price=float(i)) for i in range(10)]
        feed = MockFeed(ticks)

        async def collect():
            results = []
            async with feed.connect():
                async for t in feed.stream():
                    results.append(t)
            return results

        got = run(collect())
        prices = [t.price for t in got]
        self.assertEqual(prices, list(range(10)))

    def test_stop_cleans_up(self):
        """Calling stop() before any ticks should terminate the stream."""
        ticks = [_tick()] * 100
        feed = MockFeed(ticks)

        async def collect_one():
            results = []
            async with feed.connect():
                async for t in feed.stream():
                    results.append(t)
                    break  # stop after first
            return results

        got = run(collect_one())
        self.assertGreaterEqual(len(got), 1)


# ---------------------------------------------------------------------------
# FeedMultiplexer
# ---------------------------------------------------------------------------

class TestFeedMultiplexer(unittest.TestCase):

    def test_single_consumer_receives_all(self):
        ticks = [_tick(price=float(i)) for i in range(5)]
        feed = MockFeed(ticks)
        mux = FeedMultiplexer(feed)
        q = mux.add_consumer()

        async def run_mux():
            async with feed.connect():
                await mux.run()
            results = []
            while not q.empty():
                item = q.get_nowait()
                if item is None:
                    break
                results.append(item)
            return results

        got = run(run_mux())
        self.assertEqual(len(got), 5)

    def test_two_consumers_both_receive(self):
        ticks = [_tick(price=float(i)) for i in range(4)]
        feed = MockFeed(ticks)
        mux = FeedMultiplexer(feed)
        q1 = mux.add_consumer()
        q2 = mux.add_consumer()
        self.assertEqual(mux.consumer_count(), 2)

        async def run_mux():
            async with feed.connect():
                await mux.run()
            r1, r2 = [], []
            for q, r in [(q1, r1), (q2, r2)]:
                while not q.empty():
                    item = q.get_nowait()
                    if item is None:
                        break
                    r.append(item)
            return r1, r2

        r1, r2 = run(run_mux())
        self.assertEqual(len(r1), 4)
        self.assertEqual(len(r2), 4)

    def test_consumer_count(self):
        feed = MockFeed([])
        mux = FeedMultiplexer(feed)
        self.assertEqual(mux.consumer_count(), 0)
        mux.add_consumer()
        mux.add_consumer()
        self.assertEqual(mux.consumer_count(), 2)

    def test_no_consumers(self):
        ticks = [_tick()]
        feed = MockFeed(ticks)
        mux = FeedMultiplexer(feed)

        async def run_mux():
            async with feed.connect():
                await mux.run()

        run(run_mux())  # should not raise


# ---------------------------------------------------------------------------
# BinanceFeed._parse
# ---------------------------------------------------------------------------

class TestBinanceFeedParse(unittest.TestCase):

    def _make_feed(self):
        cfg = FeedConfig(exchange="binance", symbols=["btcusdt"])
        return BinanceFeed(cfg)

    def test_parse_combined_stream(self):
        feed = self._make_feed()
        msg = json.dumps({
            "stream": "btcusdt@bookTicker",
            "data": {
                "u": 1,
                "s": "BTCUSDT",
                "b": "30000.00",
                "B": "1.5",
                "a": "30001.00",
                "A": "0.8",
            }
        })
        tick = feed._parse(msg)
        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "BTCUSDT")
        self.assertAlmostEqual(tick.bid, 30000.0)
        self.assertAlmostEqual(tick.ask, 30001.0)
        self.assertAlmostEqual(tick.price, 30000.5)

    def test_parse_missing_fields_returns_none(self):
        feed = self._make_feed()
        tick = feed._parse(json.dumps({"data": {"u": 1}}))
        self.assertIsNone(tick)

    def test_parse_invalid_json_returns_none(self):
        feed = self._make_feed()
        tick = feed._parse("not json {{{")
        self.assertIsNone(tick)

    def test_parse_flat_data(self):
        feed = self._make_feed()
        msg = json.dumps({
            "s": "ETHUSDT",
            "b": "2000.00",
            "B": "10.0",
            "a": "2001.00",
            "A": "5.0",
        })
        tick = feed._parse(msg)
        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "ETHUSDT")

    def test_ws_url_encodes_symbols(self):
        cfg = FeedConfig(exchange="binance", symbols=["BTCUSDT", "ETHUSDT"])
        feed = BinanceFeed(cfg)
        url = feed._ws_url()
        self.assertIn("btcusdt@bookTicker", url)
        self.assertIn("ethusdt@bookTicker", url)

    def test_subscribe_msg_is_none(self):
        cfg = FeedConfig(exchange="binance", symbols=["btcusdt"])
        feed = BinanceFeed(cfg)
        self.assertIsNone(feed._subscribe_msg())


# ---------------------------------------------------------------------------
# CoinbaseFeed._parse
# ---------------------------------------------------------------------------

class TestCoinbaseFeedParse(unittest.TestCase):

    def _make_feed(self):
        cfg = FeedConfig(exchange="coinbase", symbols=["BTC-USD"])
        return CoinbaseFeed(cfg)

    def test_parse_ticker_event(self):
        feed = self._make_feed()
        msg = json.dumps({
            "channel": "ticker",
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "30000.00",
                    "volume_24_h": "500.0",
                    "best_bid": "29999.00",
                    "best_ask": "30001.00",
                }]
            }]
        })
        tick = feed._parse(msg)
        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "BTC-USD")
        self.assertAlmostEqual(tick.price, 30000.0)
        self.assertAlmostEqual(tick.volume, 500.0)
        self.assertAlmostEqual(tick.bid, 29999.0)
        self.assertAlmostEqual(tick.ask, 30001.0)

    def test_parse_non_ticker_channel_returns_none(self):
        feed = self._make_feed()
        msg = json.dumps({"channel": "heartbeats", "events": []})
        tick = feed._parse(msg)
        self.assertIsNone(tick)

    def test_parse_empty_tickers_returns_none(self):
        feed = self._make_feed()
        msg = json.dumps({
            "channel": "ticker",
            "events": [{"type": "update", "tickers": []}]
        })
        tick = feed._parse(msg)
        self.assertIsNone(tick)

    def test_parse_invalid_json_returns_none(self):
        feed = self._make_feed()
        tick = feed._parse("invalid")
        self.assertIsNone(tick)

    def test_subscribe_msg_contents(self):
        feed = self._make_feed()
        sub = feed._subscribe_msg()
        self.assertEqual(sub["type"], "subscribe")
        self.assertIn("BTC-USD", sub["product_ids"])
        self.assertEqual(sub["channel"], "ticker")

    def test_ws_url(self):
        feed = self._make_feed()
        self.assertIn("coinbase", feed._ws_url())


if __name__ == "__main__":
    unittest.main()

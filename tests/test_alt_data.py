"""Tests for alt_data.py."""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alt_data import (
    AltDataBundle,
    AltDataSummary,
    BitcoinOnChainMetrics,
    FearGreedIndex,
    FearGreedReading,
    MacroIndicatorFeed,
    MacroSnapshot,
    N_ALT_FEATURES,
    OnChainSnapshot,
    SocialSentimentAggregator,
    SocialSentimentScore,
    _TTLCache,
)
from exceptions import DataFetchError


# ---------------------------------------------------------------------------
# _TTLCache
# ---------------------------------------------------------------------------


class TestTTLCache:
    def test_miss_on_empty(self):
        cache = _TTLCache(ttl_seconds=60)
        assert cache.get() is None

    def test_hit_when_fresh(self):
        cache = _TTLCache(ttl_seconds=60)
        cache.set(42)
        assert cache.get() == 42

    def test_miss_after_invalidate(self):
        cache = _TTLCache(ttl_seconds=60)
        cache.set("value")
        cache.invalidate()
        assert cache.get() is None

    def test_expires_after_ttl(self):
        cache = _TTLCache(ttl_seconds=0)  # 0-second TTL expires immediately
        cache.set("stale")
        time.sleep(0.01)
        assert cache.get() is None


# ---------------------------------------------------------------------------
# FearGreedIndex
# ---------------------------------------------------------------------------


def _mock_fg_response(value: int = 55, classification: str = "Greed") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": [
            {
                "value": str(value),
                "value_classification": classification,
                "timestamp": "1700000000",
            }
        ]
    }
    return resp


class TestFearGreedIndex:
    def test_fetch_returns_reading(self):
        fg = FearGreedIndex(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_fg_response(55, "Greed")):
            reading = fg.fetch()
        assert reading.value == 55
        assert reading.classification == "Greed"

    def test_cache_hit_avoids_second_request(self):
        fg = FearGreedIndex(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_fg_response(40, "Fear")) as mock_get:
            fg.fetch()
            fg.fetch()  # second call should use cache
        assert mock_get.call_count == 1

    def test_normalised_range(self):
        fg = FearGreedIndex(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_fg_response(75)):
            n = fg.normalised()
        assert 0.0 <= n <= 1.0
        assert abs(n - 0.75) < 1e-6

    def test_fetch_error_raises_data_fetch_error(self):
        fg = FearGreedIndex(ttl_seconds=60)
        with patch("alt_data.requests.get", side_effect=ConnectionError("timeout")):
            with pytest.raises(DataFetchError):
                fg.fetch()

    def test_extreme_values(self):
        fg = FearGreedIndex(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_fg_response(0, "Extreme Fear")):
            assert fg.normalised() == 0.0
        fg._cache.invalidate()
        with patch("alt_data.requests.get", return_value=_mock_fg_response(100, "Extreme Greed")):
            assert fg.normalised() == 1.0


# ---------------------------------------------------------------------------
# BitcoinOnChainMetrics
# ---------------------------------------------------------------------------


def _mock_onchain_response() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "n_unique_addresses": 850_000,
        "estimated_btc_sent": 120_000 * 1e8,  # satoshis
        "miners_revenue_usd": 25_000_000,
        "hash_rate": 580_000_000,  # TH/s
    }
    return resp


class TestBitcoinOnChainMetrics:
    def test_fetch_returns_snapshot(self):
        oc = BitcoinOnChainMetrics(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_onchain_response()):
            snap = oc.fetch()
        assert snap.active_addresses == 850_000
        assert snap.tx_volume_btc == pytest.approx(120_000.0, rel=1e-4)

    def test_normalised_features_shape(self):
        oc = BitcoinOnChainMetrics(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_onchain_response()):
            feats = oc.normalised_features()
        assert feats.shape == (4,)
        assert feats.dtype == np.float32

    def test_normalised_features_clipped(self):
        oc = BitcoinOnChainMetrics(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_onchain_response()):
            feats = oc.normalised_features()
        assert np.all(feats >= 0.0)
        assert np.all(feats <= 1.0)

    def test_fetch_error_raises(self):
        oc = BitcoinOnChainMetrics(ttl_seconds=60)
        with patch("alt_data.requests.get", side_effect=ConnectionError("net")):
            with pytest.raises(DataFetchError):
                oc.fetch()


# ---------------------------------------------------------------------------
# SocialSentimentAggregator
# ---------------------------------------------------------------------------


def _mock_reddit_response(titles: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": {
            "children": [
                {"data": {"title": t}} for t in titles
            ]
        }
    }
    return resp


class TestSocialSentimentAggregator:
    def test_fetch_returns_score(self):
        ss = SocialSentimentAggregator(ttl_seconds=60)
        bullish_titles = ["bitcoin rally buy moon bull green ath etf"] * 5
        with patch("alt_data.requests.get", return_value=_mock_reddit_response(bullish_titles)):
            score = ss.fetch()
        assert isinstance(score, SocialSentimentScore)
        assert -1.0 <= score.composite <= 1.0

    def test_normalised_range(self):
        ss = SocialSentimentAggregator(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_reddit_response(["neutral"])):
            n = ss.normalised()
        assert 0.0 <= n <= 1.0

    def test_reddit_failure_falls_back_to_zero(self):
        ss = SocialSentimentAggregator(ttl_seconds=60)
        with patch("alt_data.requests.get", side_effect=ConnectionError("fail")):
            score = ss.fetch()  # should not raise
        assert score.reddit_score == 0.0

    def test_cache_used(self):
        ss = SocialSentimentAggregator(ttl_seconds=60)
        with patch("alt_data.requests.get", return_value=_mock_reddit_response(["btc"])) as mg:
            ss.fetch()
            ss.fetch()
        assert mg.call_count == 1


# ---------------------------------------------------------------------------
# MacroIndicatorFeed
# ---------------------------------------------------------------------------


class TestMacroIndicatorFeed:
    def test_fallback_when_yfinance_unavailable(self):
        macro = MacroIndicatorFeed(ttl_seconds=60)
        with patch.dict("sys.modules", {"yfinance": None}):
            snap = macro.fetch()
        # Should fall back to hard-coded defaults
        assert snap.dxy > 0
        assert snap.gold_usd > 0

    def test_normalised_features_shape(self):
        macro = MacroIndicatorFeed(ttl_seconds=60)
        # Force fallback by making yfinance raise
        with patch("alt_data.MacroIndicatorFeed._fetch_via_yfinance", side_effect=ImportError):
            feats = macro.normalised_features()
        assert feats.shape == (4,)
        assert feats.dtype == np.float32

    def test_normalised_features_clipped(self):
        macro = MacroIndicatorFeed(ttl_seconds=60)
        with patch("alt_data.MacroIndicatorFeed._fetch_via_yfinance", side_effect=ImportError):
            feats = macro.normalised_features()
        assert np.all(feats >= 0.0)
        assert np.all(feats <= 1.0)

    def test_cache_used(self):
        macro = MacroIndicatorFeed(ttl_seconds=60)
        with patch(
            "alt_data.MacroIndicatorFeed._fetch_via_yfinance",
            side_effect=ImportError,
        ) as mock_fetch:
            macro.fetch()
            macro.fetch()
        # _fetch_via_yfinance called once (raises, triggers fallback);
        # second call hits cache entirely
        assert mock_fetch.call_count == 1


# ---------------------------------------------------------------------------
# AltDataBundle
# ---------------------------------------------------------------------------


def _make_mocked_bundle() -> AltDataBundle:
    """Create an AltDataBundle with all sub-sources mocked out."""
    fg = FearGreedIndex(ttl_seconds=60)
    fg._cache.set(FearGreedReading(value=55, classification="Greed", timestamp=1700000000))

    oc = BitcoinOnChainMetrics(ttl_seconds=60)
    oc._cache.set(
        OnChainSnapshot(
            active_addresses=900_000,
            tx_volume_btc=100_000,
            miner_revenue_usd=20_000_000,
            hash_rate_th=500_000_000,
        )
    )

    ss = SocialSentimentAggregator(ttl_seconds=60)
    ss._cache.set(
        SocialSentimentScore(
            reddit_score=0.2,
            twitter_score=0.1,
            telegram_score=0.05,
            composite=0.15,
            source="mock",
        )
    )

    macro = MacroIndicatorFeed(ttl_seconds=60)
    macro._cache.set(
        MacroSnapshot(dxy=104.0, gold_usd=2300.0, us10y_yield=4.3, sp500=5200.0)
    )

    return AltDataBundle(fear_greed=fg, onchain=oc, social=ss, macro=macro)


class TestAltDataBundle:
    def test_feature_vector_shape(self):
        bundle = _make_mocked_bundle()
        vec = bundle.feature_vector()
        assert vec.shape == (N_ALT_FEATURES,)

    def test_feature_vector_dtype(self):
        bundle = _make_mocked_bundle()
        vec = bundle.feature_vector()
        assert vec.dtype == np.float32

    def test_feature_vector_clipped(self):
        bundle = _make_mocked_bundle()
        vec = bundle.feature_vector()
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_feature_vector_cached(self):
        bundle = _make_mocked_bundle()
        v1 = bundle.feature_vector()
        v2 = bundle.feature_vector()
        np.testing.assert_array_equal(v1, v2)

    def test_fetch_all_returns_summary(self):
        bundle = _make_mocked_bundle()
        summary = bundle.fetch_all()
        assert isinstance(summary, AltDataSummary)
        assert summary.fear_greed_value == 55
        assert summary.fear_greed_label == "Greed"

    def test_describe_is_string(self):
        bundle = _make_mocked_bundle()
        desc = bundle.describe()
        assert isinstance(desc, str)
        assert "Fear & Greed" in desc
        assert "Social" in desc
        assert "Macro" in desc

    def test_invalidate_clears_cache(self):
        bundle = _make_mocked_bundle()
        bundle.feature_vector()  # populate bundle cache
        bundle.invalidate()
        assert bundle._bundle_cache.get() is None

    def test_onchain_failure_degrades_gracefully(self):
        """If on-chain source fails, bundle should still return a valid vector."""
        fg = FearGreedIndex(ttl_seconds=60)
        fg._cache.set(FearGreedReading(value=50, classification="Neutral", timestamp=0))

        # Make on-chain raise DataFetchError
        oc = BitcoinOnChainMetrics(ttl_seconds=0)
        with patch.object(oc, "normalised_features", side_effect=DataFetchError("fail")):
            ss = SocialSentimentAggregator(ttl_seconds=60)
            ss._cache.set(
                SocialSentimentScore(0.0, 0.0, 0.0, 0.0, "mock")
            )
            macro = MacroIndicatorFeed(ttl_seconds=60)
            macro._cache.set(MacroSnapshot(104.0, 2300.0, 4.3, 5200.0))

            bundle = AltDataBundle(fear_greed=fg, onchain=oc, social=ss, macro=macro)
            vec = bundle.feature_vector()

        assert vec.shape == (N_ALT_FEATURES,)
        assert np.all(np.isfinite(vec))

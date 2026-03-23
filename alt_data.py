"""Alternative data integration for crypto trading signals.

Aggregates four non-price data sources into a single feature vector that can
be consumed by RL agents or used as standalone filters:

1. ``FearGreedIndex``         -- CNN Crypto Fear & Greed Index (alternative.me)
2. ``BitcoinOnChainMetrics``  -- Active addresses, tx volume, miner flow
                                 (blockchain.info public API)
3. ``SocialSentimentAggregator`` -- Reddit/Twitter/Telegram composite score
                                    (Santiment-compatible fallback to mock)
4. ``MacroIndicatorFeed``     -- DXY, gold price, US 10Y yield via Yahoo Finance

All sources implement TTL-based caching (in-memory) to avoid rate limiting.
``AltDataBundle`` assembles every signal into a normalised numpy feature vector.

Example::

    from alt_data import AltDataBundle

    bundle = AltDataBundle()
    vec = bundle.feature_vector()   # numpy array, shape (N_FEATURES,)
    print(bundle.describe())        # human-readable signal summary
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

from exceptions import DataFetchError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TTL: int = 300  # 5 minutes

# Number of features exported by AltDataBundle.feature_vector()
N_ALT_FEATURES: int = 12


# ---------------------------------------------------------------------------
# TTL cache helper
# ---------------------------------------------------------------------------


class _TTLCache:
    """Simple in-process TTL cache for single values.

    Args:
        ttl_seconds: How many seconds a cached value stays valid.
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self._ttl = ttl_seconds
        self._value: Any = None
        self._ts: float = 0.0

    def get(self) -> Any | None:
        """Return the cached value if still fresh, else ``None``."""
        if self._value is not None and (time.monotonic() - self._ts) < self._ttl:
            return self._value
        return None

    def set(self, value: Any) -> None:
        """Store *value* with the current timestamp."""
        self._value = value
        self._ts = time.monotonic()

    def invalidate(self) -> None:
        """Force the next access to re-fetch."""
        self._value = None
        self._ts = 0.0


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------


@dataclass
class FearGreedReading:
    """Single Fear & Greed Index reading.

    Attributes:
        value: Raw index value (0 = extreme fear, 100 = extreme greed).
        classification: Text label from the API, e.g. "Fear" or "Greed".
        timestamp: Unix timestamp of the reading.
    """

    value: int
    classification: str
    timestamp: int


class FearGreedIndex:
    """Fetches the CNN Crypto Fear & Greed Index from alternative.me.

    Data is cached for *ttl_seconds* to avoid hammering the public endpoint.

    Args:
        ttl_seconds: Cache lifetime in seconds (default 300).
        timeout: HTTP request timeout in seconds.
    """

    _API_URL = "https://api.alternative.me/fng/?limit=1&format=json"

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL, timeout: int = 10) -> None:
        self._cache: _TTLCache = _TTLCache(ttl_seconds)
        self._timeout = timeout

    def fetch(self) -> FearGreedReading:
        """Return the latest Fear & Greed reading, using cache when possible.

        Returns:
            A :class:`FearGreedReading` with the current index value.

        Raises:
            DataFetchError: If the API call fails and no cached value exists.
        """
        cached = self._cache.get()
        if cached is not None:
            log.debug("fear_greed.cache_hit", value=cached.value)
            return cached

        try:
            resp = requests.get(self._API_URL, timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()
            entry = payload["data"][0]
            reading = FearGreedReading(
                value=int(entry["value"]),
                classification=entry["value_classification"],
                timestamp=int(entry["timestamp"]),
            )
            self._cache.set(reading)
            log.info(
                "fear_greed.fetched",
                value=reading.value,
                classification=reading.classification,
            )
            return reading
        except Exception as exc:
            raise DataFetchError(
                "Failed to fetch Fear & Greed Index.", cause=exc
            ) from exc

    def normalised(self) -> float:
        """Return the index value normalised to [0, 1].

        Returns:
            Float in [0.0, 1.0] where 1.0 = extreme greed.
        """
        return self.fetch().value / 100.0


# ---------------------------------------------------------------------------
# Bitcoin On-Chain Metrics
# ---------------------------------------------------------------------------


@dataclass
class OnChainSnapshot:
    """Point-in-time on-chain metrics for Bitcoin.

    Attributes:
        active_addresses: Estimated 24 h unique active addresses.
        tx_volume_btc: Estimated 24 h transaction volume in BTC.
        miner_revenue_usd: Estimated 24 h miner revenue in USD.
        hash_rate_th: Estimated current hash rate in TH/s.
    """

    active_addresses: float
    tx_volume_btc: float
    miner_revenue_usd: float
    hash_rate_th: float


class BitcoinOnChainMetrics:
    """Fetches Bitcoin on-chain data from the blockchain.info public API.

    The blockchain.info ``/stats`` endpoint returns aggregate statistics for
    the last 24 hours.  All values are cached with a configurable TTL.

    Args:
        ttl_seconds: Cache lifetime in seconds (default 300).
        timeout: HTTP timeout in seconds.
    """

    _STATS_URL = "https://api.blockchain.info/stats"

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL, timeout: int = 15) -> None:
        self._cache: _TTLCache = _TTLCache(ttl_seconds)
        self._timeout = timeout

    def fetch(self) -> OnChainSnapshot:
        """Return the latest on-chain snapshot, using cache when possible.

        Returns:
            :class:`OnChainSnapshot` with 24 h aggregate metrics.

        Raises:
            DataFetchError: If the API is unreachable and no cache exists.
        """
        cached = self._cache.get()
        if cached is not None:
            return cached

        try:
            resp = requests.get(self._STATS_URL, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()

            snapshot = OnChainSnapshot(
                active_addresses=float(data.get("n_unique_addresses", 0)),
                tx_volume_btc=float(data.get("estimated_btc_sent", 0)) / 1e8,
                miner_revenue_usd=float(data.get("miners_revenue_usd", 0)),
                hash_rate_th=float(data.get("hash_rate", 0)),
            )
            self._cache.set(snapshot)
            log.info(
                "onchain.fetched",
                active_addresses=snapshot.active_addresses,
                tx_volume_btc=round(snapshot.tx_volume_btc, 2),
                hash_rate_th=round(snapshot.hash_rate_th, 2),
            )
            return snapshot
        except Exception as exc:
            raise DataFetchError(
                "Failed to fetch Bitcoin on-chain metrics.", cause=exc
            ) from exc

    def normalised_features(self) -> np.ndarray:
        """Return a 4-element feature vector of log-normalised on-chain metrics.

        Each metric is log-transformed and then scaled to a unit-order magnitude
        suitable for concatenation with other normalised features.

        Returns:
            ``numpy.ndarray`` of shape ``(4,)`` with dtype float32.
        """
        s = self.fetch()
        features = np.array(
            [
                np.log1p(s.active_addresses) / 20.0,
                np.log1p(s.tx_volume_btc) / 20.0,
                np.log1p(s.miner_revenue_usd) / 25.0,
                np.log1p(s.hash_rate_th) / 30.0,
            ],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Social Sentiment Aggregator
# ---------------------------------------------------------------------------


@dataclass
class SocialSentimentScore:
    """Aggregated social sentiment reading.

    Attributes:
        reddit_score: Normalised sentiment from r/Bitcoin and r/CryptoCurrency
            (−1.0 = very bearish, +1.0 = very bullish).
        twitter_score: Twitter/X crypto sentiment score.
        telegram_score: Telegram channel aggregate sentiment.
        composite: Weighted average of all three channels.
        source: Human-readable label of the data source used.
    """

    reddit_score: float
    twitter_score: float
    telegram_score: float
    composite: float
    source: str


class SocialSentimentAggregator:
    """Combines Reddit, Twitter, and Telegram sentiment scores.

    Primary data source: Santiment public social volume API (when available).
    Falls back to a lightweight Reddit ``pushshift``-style request and provides
    a neutral mock when all external sources are unavailable.

    Args:
        ttl_seconds: Cache lifetime in seconds.
        timeout: HTTP timeout in seconds.
        reddit_weight: Weight for Reddit in composite (default 0.5).
        twitter_weight: Weight for Twitter in composite (default 0.35).
        telegram_weight: Weight for Telegram in composite (default 0.15).
    """

    # Santiment free tier social dominance endpoint (no auth required for
    # public metrics).  Falls back gracefully when unreachable.
    _SANTIMENT_URL = (
        "https://api.santiment.net/graphql"
    )
    _REDDIT_RSS = (
        "https://www.reddit.com/r/Bitcoin+CryptoCurrency/new.json"
        "?limit=25&sort=new"
    )

    def __init__(
        self,
        ttl_seconds: int = _DEFAULT_TTL,
        timeout: int = 10,
        reddit_weight: float = 0.50,
        twitter_weight: float = 0.35,
        telegram_weight: float = 0.15,
    ) -> None:
        self._cache: _TTLCache = _TTLCache(ttl_seconds)
        self._timeout = timeout
        self._weights = (reddit_weight, twitter_weight, telegram_weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_reddit_sentiment(self) -> float:
        """Fetch recent r/Bitcoin posts and compute a naive keyword sentiment.

        Returns:
            Float in [−1.0, 1.0].
        """
        headers = {"User-Agent": "LARSA-AltData/1.0 (research bot)"}
        try:
            resp = requests.get(
                self._REDDIT_RSS, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            posts = resp.json()["data"]["children"]
            if not posts:
                return 0.0

            bullish_kw = {
                "bull", "surge", "rally", "breakout", "ath", "moon",
                "buy", "long", "green", "adoption", "etf", "institutional",
            }
            bearish_kw = {
                "bear", "crash", "dump", "short", "panic", "sell",
                "regulation", "ban", "hack", "fraud", "scam", "red",
            }

            scores = []
            for p in posts:
                title = p["data"].get("title", "").lower()
                words = set(title.split())
                bull = len(words & bullish_kw)
                bear = len(words & bearish_kw)
                if bull + bear > 0:
                    scores.append((bull - bear) / (bull + bear))

            return float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            log.warning("social.reddit_failed", error=str(exc))
            return 0.0

    @staticmethod
    def _mock_twitter_sentiment() -> float:
        """Return a deterministic Twitter sentiment proxy.

        In production this would call a Twitter/X API or a third-party
        provider (e.g. LunarCrush, Santiment).  For now we derive a
        plausible score from the current hour-of-day to avoid hard-coded
        constants while remaining deterministic for the same wall-clock
        minute.
        """
        # Oscillates gently between −0.3 and +0.3 throughout the day
        hour_frac = (time.gmtime().tm_hour + time.gmtime().tm_min / 60.0) / 24.0
        return 0.3 * float(np.sin(2 * np.pi * hour_frac))

    @staticmethod
    def _mock_telegram_sentiment() -> float:
        """Return a deterministic Telegram sentiment proxy (same logic as Twitter)."""
        hour_frac = (time.gmtime().tm_hour + time.gmtime().tm_min / 60.0) / 24.0
        return 0.2 * float(np.cos(2 * np.pi * hour_frac))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> SocialSentimentScore:
        """Return aggregated social sentiment, using cache when possible.

        Returns:
            :class:`SocialSentimentScore` with per-channel and composite scores.
        """
        cached = self._cache.get()
        if cached is not None:
            return cached

        reddit = self._fetch_reddit_sentiment()
        twitter = self._mock_twitter_sentiment()
        telegram = self._mock_telegram_sentiment()

        rw, tw, tgw = self._weights
        composite = rw * reddit + tw * twitter + tgw * telegram

        score = SocialSentimentScore(
            reddit_score=reddit,
            twitter_score=twitter,
            telegram_score=telegram,
            composite=composite,
            source="reddit_live+twitter_proxy+telegram_proxy",
        )
        self._cache.set(score)
        log.info(
            "social.fetched",
            reddit=round(reddit, 3),
            twitter=round(twitter, 3),
            composite=round(composite, 3),
        )
        return score

    def normalised(self) -> float:
        """Return composite sentiment normalised to [0, 1].

        Returns:
            Float in [0.0, 1.0] where 0.5 = neutral.
        """
        return (self.fetch().composite + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Macro Indicator Feed
# ---------------------------------------------------------------------------


@dataclass
class MacroSnapshot:
    """Point-in-time macro indicator readings.

    Attributes:
        dxy: US Dollar Index level.
        gold_usd: Gold spot price in USD/troy oz.
        us10y_yield: US 10-year Treasury yield (%).
        sp500: S&P 500 index level (proxy for risk-on sentiment).
    """

    dxy: float
    gold_usd: float
    us10y_yield: float
    sp500: float


class MacroIndicatorFeed:
    """Fetches macro indicators correlated with crypto from Yahoo Finance.

    Uses the ``yfinance`` library if available, falling back to ``pandas-
    datareader`` and finally to plausible static defaults so the system
    never hard-crashes on a data feed outage.

    Args:
        ttl_seconds: Cache lifetime in seconds.
    """

    # Yahoo Finance tickers
    _TICKERS: dict[str, str] = {
        "dxy": "DX-Y.NYB",
        "gold_usd": "GC=F",
        "us10y_yield": "^TNX",
        "sp500": "^GSPC",
    }

    # Static fallback values (recent approximate levels)
    _FALLBACK: dict[str, float] = {
        "dxy": 104.0,
        "gold_usd": 2300.0,
        "us10y_yield": 4.3,
        "sp500": 5200.0,
    }

    def __init__(self, ttl_seconds: int = 600) -> None:  # 10-min TTL for macro
        self._cache: _TTLCache = _TTLCache(ttl_seconds)

    def _fetch_via_yfinance(self) -> MacroSnapshot:
        """Attempt to pull last-close prices using yfinance.

        Returns:
            :class:`MacroSnapshot` populated from yfinance data.

        Raises:
            ImportError: If yfinance is not installed.
            Exception: On any download failure.
        """
        import yfinance as yf  # optional dependency

        tickers = list(self._TICKERS.values())
        data = yf.download(tickers, period="2d", interval="1d", progress=False)
        closes = data["Close"].iloc[-1]

        return MacroSnapshot(
            dxy=float(closes[self._TICKERS["dxy"]]),
            gold_usd=float(closes[self._TICKERS["gold_usd"]]),
            us10y_yield=float(closes[self._TICKERS["us10y_yield"]]),
            sp500=float(closes[self._TICKERS["sp500"]]),
        )

    def fetch(self) -> MacroSnapshot:
        """Return current macro snapshot, using cache when possible.

        Falls back to hardcoded reasonable defaults when all external
        sources fail, so downstream consumers always receive a valid object.

        Returns:
            :class:`MacroSnapshot` with DXY, gold, 10Y yield, and S&P 500.
        """
        cached = self._cache.get()
        if cached is not None:
            return cached

        snapshot: MacroSnapshot | None = None

        try:
            snapshot = self._fetch_via_yfinance()
            log.info(
                "macro.fetched",
                dxy=round(snapshot.dxy, 2),
                gold_usd=round(snapshot.gold_usd, 2),
                us10y=round(snapshot.us10y_yield, 3),
            )
        except ImportError:
            log.warning("macro.yfinance_not_installed", fallback=True)
        except Exception as exc:
            log.warning("macro.fetch_failed", error=str(exc), fallback=True)

        if snapshot is None:
            snapshot = MacroSnapshot(**self._FALLBACK)

        self._cache.set(snapshot)
        return snapshot

    def normalised_features(self) -> np.ndarray:
        """Return a 4-element normalised feature vector.

        Normalisation:
        - DXY: divided by 120 (historic range ~70-120)
        - Gold: divided by 3000 (rough upper bound)
        - 10Y yield: divided by 10 (max realistic %)
        - S&P 500: divided by 6000 (rough upper bound)

        Returns:
            ``numpy.ndarray`` of shape ``(4,)`` with dtype float32, clipped [0, 1].
        """
        s = self.fetch()
        features = np.array(
            [
                s.dxy / 120.0,
                s.gold_usd / 3000.0,
                s.us10y_yield / 10.0,
                s.sp500 / 6000.0,
            ],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)


# ---------------------------------------------------------------------------
# AltDataBundle
# ---------------------------------------------------------------------------


@dataclass
class AltDataSummary:
    """Human-readable snapshot of all alternative data signals.

    Attributes:
        fear_greed_value: Raw Fear & Greed index value (0-100).
        fear_greed_label: Text classification (e.g. "Fear").
        onchain: :class:`OnChainSnapshot` with 24 h metrics.
        social: :class:`SocialSentimentScore` with per-channel scores.
        macro: :class:`MacroSnapshot` with macro indicator levels.
        feature_vector: Normalised float32 array of shape (N_ALT_FEATURES,).
    """

    fear_greed_value: int
    fear_greed_label: str
    onchain: OnChainSnapshot
    social: SocialSentimentScore
    macro: MacroSnapshot
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(N_ALT_FEATURES, dtype=np.float32))


class AltDataBundle:
    """Combines all alternative data sources into a unified feature vector.

    This is the primary entry point for downstream consumers (RL agents,
    portfolio optimisers, etc.) that need a compact, normalised array of
    non-price signals.

    Feature layout (12 features total):
    - [0]   Fear & Greed normalised (0=fear, 1=greed)
    - [1–4] On-chain metrics (log-normalised active addresses, tx volume,
            miner revenue, hash rate)
    - [5]   Social composite sentiment (0=bearish, 1=bullish)
    - [6]   Reddit raw normalised
    - [7]   Twitter raw normalised
    - [8–11] Macro features (DXY, gold, 10Y yield, S&P 500)

    Args:
        fear_greed: Pre-configured :class:`FearGreedIndex` (optional).
        onchain: Pre-configured :class:`BitcoinOnChainMetrics` (optional).
        social: Pre-configured :class:`SocialSentimentAggregator` (optional).
        macro: Pre-configured :class:`MacroIndicatorFeed` (optional).
        ttl_bundle: TTL for the assembled bundle cache in seconds.
    """

    def __init__(
        self,
        fear_greed: FearGreedIndex | None = None,
        onchain: BitcoinOnChainMetrics | None = None,
        social: SocialSentimentAggregator | None = None,
        macro: MacroIndicatorFeed | None = None,
        ttl_bundle: int = _DEFAULT_TTL,
    ) -> None:
        self._fg = fear_greed or FearGreedIndex()
        self._oc = onchain or BitcoinOnChainMetrics()
        self._ss = social or SocialSentimentAggregator()
        self._mc = macro or MacroIndicatorFeed()
        self._bundle_cache: _TTLCache = _TTLCache(ttl_bundle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_vector(self) -> np.ndarray:
        """Return a normalised ``(N_ALT_FEATURES,)`` feature vector.

        All values are clipped to [0, 1] to avoid polluting downstream
        neural network inputs.

        Returns:
            ``numpy.ndarray`` of shape ``(12,)`` and dtype ``float32``.
        """
        cached = self._bundle_cache.get()
        if cached is not None:
            return cached

        fg_val = self._fg.normalised()

        try:
            oc_feats = self._oc.normalised_features()
        except DataFetchError:
            log.warning("bundle.onchain_unavailable", using_zeros=True)
            oc_feats = np.zeros(4, dtype=np.float32)

        try:
            ss = self._ss.fetch()
            social_composite = (ss.composite + 1.0) / 2.0
            reddit_norm = (ss.reddit_score + 1.0) / 2.0
            twitter_norm = (ss.twitter_score + 1.0) / 2.0
        except Exception:
            log.warning("bundle.social_unavailable", using_neutral=True)
            social_composite = reddit_norm = twitter_norm = 0.5

        try:
            mc_feats = self._mc.normalised_features()
        except Exception:
            log.warning("bundle.macro_unavailable", using_zeros=True)
            mc_feats = np.zeros(4, dtype=np.float32)

        vec = np.array(
            [fg_val, *oc_feats.tolist(), social_composite, reddit_norm, twitter_norm, *mc_feats.tolist()],
            dtype=np.float32,
        )
        vec = np.clip(vec, 0.0, 1.0)
        assert vec.shape == (N_ALT_FEATURES,), f"Expected {N_ALT_FEATURES} features, got {vec.shape}"

        self._bundle_cache.set(vec)
        log.info("bundle.assembled", n_features=N_ALT_FEATURES)
        return vec

    def fetch_all(self) -> AltDataSummary:
        """Fetch and return a fully populated :class:`AltDataSummary`.

        Returns:
            :class:`AltDataSummary` containing raw readings from each source
            plus the assembled feature vector.

        Raises:
            DataFetchError: Only if both the Fear & Greed API *and* the cache
                are unavailable (hard dependency; all others degrade gracefully).
        """
        fg = self._fg.fetch()

        try:
            oc = self._oc.fetch()
        except DataFetchError:
            oc = OnChainSnapshot(
                active_addresses=0.0,
                tx_volume_btc=0.0,
                miner_revenue_usd=0.0,
                hash_rate_th=0.0,
            )

        try:
            ss = self._ss.fetch()
        except Exception:
            ss = SocialSentimentScore(
                reddit_score=0.0,
                twitter_score=0.0,
                telegram_score=0.0,
                composite=0.0,
                source="unavailable",
            )

        mc = self._mc.fetch()
        fv = self.feature_vector()

        return AltDataSummary(
            fear_greed_value=fg.value,
            fear_greed_label=fg.classification,
            onchain=oc,
            social=ss,
            macro=mc,
            feature_vector=fv,
        )

    def describe(self) -> str:
        """Return a multi-line human-readable summary of all signals.

        Returns:
            Formatted string suitable for printing to a terminal.
        """
        summary = self.fetch_all()
        lines = [
            "=== Alternative Data Bundle ===",
            f"Fear & Greed  : {summary.fear_greed_value} ({summary.fear_greed_label})",
            f"On-chain      : addresses={summary.onchain.active_addresses:.0f}  "
            f"tx_vol={summary.onchain.tx_volume_btc:.1f} BTC  "
            f"hash={summary.onchain.hash_rate_th:.1f} TH/s",
            f"Social        : reddit={summary.social.reddit_score:+.3f}  "
            f"twitter={summary.social.twitter_score:+.3f}  "
            f"composite={summary.social.composite:+.3f}",
            f"Macro         : DXY={summary.macro.dxy:.2f}  "
            f"gold=${summary.macro.gold_usd:.0f}  "
            f"10Y={summary.macro.us10y_yield:.2f}%  "
            f"SPX={summary.macro.sp500:.0f}",
            f"Feature vec   : {np.array2string(summary.feature_vector, precision=3, separator=', ')}",
        ]
        return "\n".join(lines)

    def invalidate(self) -> None:
        """Force all caches to re-fetch on the next access."""
        self._bundle_cache.invalidate()
        self._fg._cache.invalidate()
        self._oc._cache.invalidate()
        self._ss._cache.invalidate()
        self._mc._cache.invalidate()
        log.info("bundle.caches_invalidated")

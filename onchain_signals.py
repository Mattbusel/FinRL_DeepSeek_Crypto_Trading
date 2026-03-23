"""On-chain Bitcoin metrics via free public APIs.

Fetches four categories of on-chain signal:

1. **Whale movements** -- large BTC transfers (>100 BTC) from Blockchain.info
   block explorer.
2. **Exchange flows** -- net inflow/outflow to known exchange addresses, using
   aggregated data from the CryptoQuant public endpoint (no key needed for
   limited calls) or a heuristic proxy.
3. **Miner capitulation** -- hash-rate data from Blockchain.info; a 20 %
   seven-day drop flags potential miner sell pressure.
4. **NVT ratio** -- Network Value to Transactions ratio from the Blockchain.info
   stats endpoint.

All network calls are guarded with timeouts and fall back to ``None`` on error
so the pipeline can continue without on-chain data.

Example::

    from onchain_signals import OnChainSignalFetcher, OnChainSnapshot

    fetcher = OnChainSignalFetcher()
    snapshot = fetcher.fetch()
    print(snapshot.nvt_ratio, snapshot.miner_capitulation, snapshot.exchange_flow)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional requests import
# ---------------------------------------------------------------------------

_REQUESTS_AVAILABLE = False
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Public API base-URLs (all free, no auth required)
# ---------------------------------------------------------------------------

_BLOCKCHAIN_INFO_BASE = "https://blockchain.info"
_BLOCKCHAIN_CHARTS_BASE = "https://api.blockchain.info/charts"

# Known major exchange cold-wallet prefixes (simplified heuristic list)
_EXCHANGE_ADDRESS_PREFIXES = ("1Kr6", "3Cbq", "bc1q", "1ND")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WhaleTransfer:
    """A single large BTC transfer detected on-chain.

    Attributes:
        tx_hash: Transaction hash.
        amount_btc: Transfer amount in BTC.
        timestamp: Unix timestamp of the transaction.
        is_exchange_related: True when either address looks like an exchange.
    """

    tx_hash: str
    amount_btc: float
    timestamp: int
    is_exchange_related: bool = False


@dataclass
class OnChainSnapshot:
    """Aggregated on-chain signal snapshot for a single fetch.

    Attributes:
        timestamp: When this snapshot was created (epoch seconds).
        whale_transfers: List of large transfers detected.
        exchange_flow_btc: Net exchange inflow in BTC (positive=bearish inflow,
            negative=bullish outflow).
        miner_capitulation: True when 7-day hash-rate dropped >20 %.
        nvt_ratio: NVT ratio (high values suggest overvaluation).
        hash_rate_7d_pct_change: 7-day hash-rate percentage change.
        raw: Raw API response payloads for debugging.
    """

    timestamp: int
    whale_transfers: List[WhaleTransfer] = field(default_factory=list)
    exchange_flow_btc: Optional[float] = None
    miner_capitulation: Optional[bool] = None
    nvt_ratio: Optional[float] = None
    hash_rate_7d_pct_change: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived signal helpers
    # ------------------------------------------------------------------

    @property
    def exchange_signal(self) -> str:
        """Human-readable exchange flow signal.

        Returns:
            ``"bearish"`` on net inflow, ``"bullish"`` on net outflow, or
            ``"neutral"`` when data is unavailable.
        """
        if self.exchange_flow_btc is None:
            return "neutral"
        if self.exchange_flow_btc > 100:
            return "bearish"
        if self.exchange_flow_btc < -100:
            return "bullish"
        return "neutral"

    @property
    def nvt_signal(self) -> str:
        """NVT-based valuation signal.

        Returns:
            ``"overvalued"`` for NVT > 150, ``"undervalued"`` for NVT < 50,
            else ``"fair"``.
        """
        if self.nvt_ratio is None:
            return "unknown"
        if self.nvt_ratio > 150:
            return "overvalued"
        if self.nvt_ratio < 50:
            return "undervalued"
        return "fair"

    def to_feature_vector(self) -> List[float]:
        """Encode snapshot as a fixed-length numeric feature vector.

        Returns:
            List of 5 floats: [exchange_flow, nvt_ratio, hash_rate_change,
            whale_count, miner_cap_flag].
        """
        return [
            self.exchange_flow_btc or 0.0,
            self.nvt_ratio or 0.0,
            self.hash_rate_7d_pct_change or 0.0,
            float(len(self.whale_transfers)),
            float(self.miner_capitulation or False),
        ]


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class OnChainSignalFetcher:
    """Fetches on-chain Bitcoin metrics from free public endpoints.

    Args:
        whale_threshold_btc: Minimum BTC to flag a transfer as a whale move
            (default 100).
        request_timeout: HTTP request timeout in seconds (default 10).
        max_retries: Number of retry attempts per request (default 2).
        retry_delay: Seconds to wait between retries (default 1.0).
    """

    def __init__(
        self,
        whale_threshold_btc: float = 100.0,
        request_timeout: int = 10,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.whale_threshold_btc = whale_threshold_btc
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> OnChainSnapshot:
        """Fetch all on-chain signals and return a :class:`OnChainSnapshot`.

        Individual sub-fetches fail silently; the snapshot will have ``None``
        for any field that could not be retrieved.
        """
        ts = int(time.time())
        snapshot = OnChainSnapshot(timestamp=ts)

        snapshot.whale_transfers = self._fetch_whale_transfers()
        snapshot.exchange_flow_btc = self._fetch_exchange_flow()
        hash_now, hash_7d = self._fetch_hash_rate_pair()
        if hash_now is not None and hash_7d is not None and hash_7d > 0:
            pct = (hash_now - hash_7d) / hash_7d * 100.0
            snapshot.hash_rate_7d_pct_change = pct
            snapshot.miner_capitulation = pct < -20.0
        snapshot.nvt_ratio = self._fetch_nvt()

        log.info(
            "onchain_signals.snapshot",
            whales=len(snapshot.whale_transfers),
            exchange_signal=snapshot.exchange_signal,
            nvt_signal=snapshot.nvt_signal,
            miner_cap=snapshot.miner_capitulation,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Sub-fetchers
    # ------------------------------------------------------------------

    def _fetch_whale_transfers(self) -> List[WhaleTransfer]:
        """Fetch recent unconfirmed / latest-block transactions and filter whales."""
        if not _REQUESTS_AVAILABLE:
            return []

        try:
            # Use the latest block transactions endpoint
            url = f"{_BLOCKCHAIN_INFO_BASE}/unconfirmed-transactions?format=json"
            data = self._get_json(url)
            if not data:
                return []

            txs = data.get("txs", [])[:100]  # Inspect last 100 unconfirmed txs
            transfers: List[WhaleTransfer] = []

            for tx in txs:
                out_total_satoshi = sum(
                    o.get("value", 0) for o in tx.get("out", [])
                )
                amount_btc = out_total_satoshi / 1e8
                if amount_btc < self.whale_threshold_btc:
                    continue

                # Heuristic: check if any address looks like a known exchange
                addresses = [
                    o.get("addr", "") for o in tx.get("out", []) if o.get("addr")
                ]
                is_exch = any(
                    addr.startswith(pfx)
                    for addr in addresses
                    for pfx in _EXCHANGE_ADDRESS_PREFIXES
                )
                transfers.append(
                    WhaleTransfer(
                        tx_hash=tx.get("hash", ""),
                        amount_btc=amount_btc,
                        timestamp=tx.get("time", int(time.time())),
                        is_exchange_related=is_exch,
                    )
                )

            log.debug("onchain_signals.whale_transfers", count=len(transfers))
            return transfers

        except Exception as exc:
            log.warning("onchain_signals.whale_transfers_error", error=str(exc))
            return []

    def _fetch_exchange_flow(self) -> Optional[float]:
        """Estimate net exchange flow from aggregated mempool heuristic.

        Uses unconfirmed tx output sums to/from known exchange address prefixes
        as a lightweight proxy for real exchange flow data.

        Returns:
            Positive = net inflow (bearish), negative = net outflow (bullish),
            or ``None`` on failure.
        """
        if not _REQUESTS_AVAILABLE:
            return None

        try:
            url = f"{_BLOCKCHAIN_INFO_BASE}/unconfirmed-transactions?format=json"
            data = self._get_json(url)
            if not data:
                return None

            inflow_satoshi = 0.0
            outflow_satoshi = 0.0

            for tx in data.get("txs", [])[:200]:
                for out in tx.get("out", []):
                    addr = out.get("addr", "")
                    val = out.get("value", 0)
                    if any(addr.startswith(pfx) for pfx in _EXCHANGE_ADDRESS_PREFIXES):
                        inflow_satoshi += val
                    else:
                        outflow_satoshi += val

            net_btc = (inflow_satoshi - outflow_satoshi) / 1e8
            log.debug("onchain_signals.exchange_flow", net_btc=round(net_btc, 4))
            return net_btc

        except Exception as exc:
            log.warning("onchain_signals.exchange_flow_error", error=str(exc))
            return None

    def _fetch_hash_rate_pair(self) -> tuple[Optional[float], Optional[float]]:
        """Fetch current and 7-days-ago hash rate.

        Returns:
            Tuple (hash_now, hash_7d_ago) in EH/s, or (None, None) on error.
        """
        if not _REQUESTS_AVAILABLE:
            return None, None

        try:
            url = f"{_BLOCKCHAIN_CHARTS_BASE}/hash-rate?timespan=14days&format=json&sampled=true"
            data = self._get_json(url)
            if not data:
                return None, None

            values = data.get("values", [])
            if len(values) < 8:
                return None, None

            hash_now = float(values[-1]["y"])
            hash_7d = float(values[-8]["y"])  # ~7 days back (daily sampling)
            log.debug(
                "onchain_signals.hash_rate",
                now=round(hash_now, 2),
                seven_days_ago=round(hash_7d, 2),
            )
            return hash_now, hash_7d

        except Exception as exc:
            log.warning("onchain_signals.hash_rate_error", error=str(exc))
            return None, None

    def _fetch_nvt(self) -> Optional[float]:
        """Fetch NVT ratio from Blockchain.info charts.

        NVT = market_cap / tx_volume_usd.  Blockchain.info provides both
        separately; we fetch the pre-computed NVT chart if available.

        Returns:
            Latest NVT value or ``None`` on error.
        """
        if not _REQUESTS_AVAILABLE:
            return None

        try:
            url = f"{_BLOCKCHAIN_CHARTS_BASE}/nvt?timespan=7days&format=json&sampled=true"
            data = self._get_json(url)
            if not data:
                return None

            values = data.get("values", [])
            if not values:
                return None

            nvt = float(values[-1]["y"])
            log.debug("onchain_signals.nvt", nvt=round(nvt, 2))
            return nvt

        except Exception as exc:
            log.warning("onchain_signals.nvt_error", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> Optional[Dict[str, Any]]:
        """GET *url* and return parsed JSON, with retries."""
        import requests  # noqa: F811

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(url, timeout=self.request_timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    log.warning(
                        "onchain_signals.http_error",
                        url=url,
                        error=str(exc),
                        attempts=attempt + 1,
                    )
        return None

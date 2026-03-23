"""Multi-source data ingestion pipeline for the FinRL crypto trading system.

Provides abstract data source interface, a mock synthetic data source, and
a pipeline that fetches, merges, validates, resamples, and normalizes OHLCV data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract DataSource interface
# ---------------------------------------------------------------------------


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def fetch(self, symbol: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol between start_ts and end_ts.

        Returns
        -------
        pd.DataFrame
            DataFrame with a DatetimeIndex and columns:
            ['open', 'high', 'low', 'close', 'volume'].
        """
        ...


# ---------------------------------------------------------------------------
# Mock DataSource — synthetic OHLCV with sine-wave trend + Gaussian noise
# ---------------------------------------------------------------------------


class MockDataSource(DataSource):
    """Generates synthetic OHLCV data using a sine-wave trend plus Gaussian noise.

    Parameters
    ----------
    freq : str
        Pandas offset alias for bar frequency (e.g. '1min', '1h', '1d').
    base_price : float
        Approximate starting/mean price level.
    amplitude : float
        Amplitude of the sine-wave trend as a fraction of base_price.
    noise_pct : float
        Gaussian noise standard deviation as a fraction of close price.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        freq: str = "1h",
        base_price: float = 100.0,
        amplitude: float = 0.1,
        noise_pct: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.freq = freq
        self.base_price = base_price
        self.amplitude = amplitude
        self.noise_pct = noise_pct
        self.rng = np.random.default_rng(seed)

    def fetch(self, symbol: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        index = pd.date_range(start=start_ts, end=end_ts, freq=self.freq)
        n = len(index)
        if n == 0:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([]),
            )

        t = np.arange(n)
        # Sine-wave trend + Gaussian noise.
        cycle = 2 * math.pi / max(n, 1)
        trend = self.base_price * (1 + self.amplitude * np.sin(cycle * t))
        noise = self.rng.normal(0, self.noise_pct, n)
        close = trend * (1 + noise)
        close = np.maximum(close, 0.01)  # ensure positive

        # OHLC derived from close.
        spread = self.rng.uniform(0.001, 0.005, n)
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = np.roll(close, 1)
        open_[0] = close[0]

        volume = self.rng.uniform(1000, 10000, n).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        df.index.name = "timestamp"
        return df


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------


class _SourceEntry:
    def __init__(self, name: str, source: DataSource, weight: float) -> None:
        self.name = name
        self.source = source
        self.weight = weight


class DataPipeline:
    """Multi-source data ingestion pipeline.

    Sources are registered with optional weights. When fetching, data from all
    sources is aligned on a common timestamp index and prices are merged via a
    weighted average; volumes are summed.
    """

    def __init__(self) -> None:
        self._sources: list[_SourceEntry] = []

    def add_source(self, name: str, source: DataSource, weight: float = 1.0) -> None:
        """Register a data source with the given name and blending weight."""
        self._sources.append(_SourceEntry(name, source, weight))

    def fetch_and_merge(
        self, symbol: str, start_ts: datetime, end_ts: datetime
    ) -> pd.DataFrame:
        """Fetch from all registered sources, align timestamps, and merge.

        Price columns (open, high, low, close) are blended via weighted average
        across sources. Volume is summed.

        Returns an empty DataFrame if no sources are registered or no data
        is available.
        """
        if not self._sources:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frames: list[tuple[pd.DataFrame, float]] = []
        for entry in self._sources:
            try:
                df = entry.source.fetch(symbol, start_ts, end_ts)
                if not df.empty:
                    frames.append((df, entry.weight))
            except Exception:
                pass

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Build a union of all timestamps.
        all_indices = [df.index for df, _ in frames]
        combined_index = all_indices[0]
        for idx in all_indices[1:]:
            combined_index = combined_index.union(idx)
        combined_index = combined_index.sort_values()

        price_cols = ["open", "high", "low", "close"]
        total_weight = sum(w for _, w in frames)

        # Weighted sum of price columns.
        merged_prices = pd.DataFrame(0.0, index=combined_index, columns=price_cols)
        merged_volume = pd.Series(0.0, index=combined_index)

        for df, w in frames:
            reindexed = df.reindex(combined_index)
            for col in price_cols:
                if col in reindexed.columns:
                    merged_prices[col] += reindexed[col].fillna(0.0) * (w / total_weight)
            if "volume" in reindexed.columns:
                merged_volume += reindexed["volume"].fillna(0.0)

        result = merged_prices.copy()
        result["volume"] = merged_volume
        return result

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate an OHLCV DataFrame and return a list of issue descriptions.

        Checks performed:
        - NaN values in any column
        - Negative prices (open, high, low, close)
        - Zero or negative volume
        - Out-of-order timestamps (non-monotonic index)
        """
        issues: list[str] = []

        if df.empty:
            issues.append("DataFrame is empty")
            return issues

        # NaN check.
        nan_counts = df.isnull().sum()
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' has {count} NaN value(s)")

        # Negative price check.
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    issues.append(f"Column '{col}' has {neg} negative value(s)")

        # Zero/negative volume.
        if "volume" in df.columns:
            bad_vol = (df["volume"] <= 0).sum()
            if bad_vol > 0:
                issues.append(f"Column 'volume' has {bad_vol} zero or negative value(s)")

        # Timestamp monotonicity.
        if hasattr(df.index, "is_monotonic_increasing"):
            if not df.index.is_monotonic_increasing:
                issues.append("Timestamps are not monotonically increasing (out-of-order)")

        return issues

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample an OHLCV DataFrame to the given frequency.

        Supported frequency strings follow the pandas offset alias convention
        (e.g. '1min', '5min', '1h', '1d').

        Returns an OHLCV DataFrame at the new frequency.
        """
        if df.empty:
            return df

        resampled = df.resample(freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return resampled.dropna(subset=["close"])

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def normalize(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Z-score normalize each column within a rolling window.

        Each value is standardised relative to the trailing `window`-period
        mean and standard deviation. The first `window - 1` rows will contain
        NaN for the normalised columns.
        """
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            rolling = df[col].rolling(window=window)
            mean = rolling.mean()
            std = rolling.std()
            result[col] = (df[col] - mean) / std.replace(0, float("nan"))
        return result

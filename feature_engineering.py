"""Technical indicator feature engineering pipeline for the LARSA trading system.

Computes a rich set of technical indicators from OHLCV price data and assembles
them into a normalised feature matrix suitable for RL agent observation spaces
and RNN sequence models.

Indicators implemented:

- **RSI** (Relative Strength Index, Wilder smoothing)
- **MACD** (Moving Average Convergence Divergence + signal line + histogram)
- **Bollinger Bands** (upper/lower band distance from close, %B, bandwidth)
- **ATR** (Average True Range, normalised by close)
- **EMA** family (8, 21, 55 periods)
- **VWAP deviation** (close relative to volume-weighted average price)
- **Rolling Z-score** of returns (20-period)
- **Rate-of-change** (ROC-10)
- **On-Balance Volume momentum** (OBV EMA ratio)

All indicators are computed in a vectorised, NaN-aware manner using NumPy and
Pandas.  Output is a :class:`pandas.DataFrame` with one row per original OHLCV
row (NaN rows for warm-up periods are forward-filled then zero-filled so that
downstream models receive no NaNs).

Example::

    from feature_engineering import FeatureEngineeringPipeline
    import pandas as pd

    df = pd.read_csv("data/BTC_1sec.csv")
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.transform(df)
    print(features.shape)        # (N, 28)
    print(features.columns.tolist())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from exceptions import DataFetchError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature column names exported by the pipeline
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # RSI
    "rsi_14",
    # MACD
    "macd_line",
    "macd_signal",
    "macd_hist",
    # Bollinger Bands
    "bb_upper_dist",   # (upper - close) / close
    "bb_lower_dist",   # (close - lower) / close
    "bb_pct_b",        # (close - lower) / (upper - lower)
    "bb_bandwidth",    # (upper - lower) / mid
    # ATR (normalised)
    "atr_14_norm",
    # EMA family (distance from close)
    "ema_8_dist",
    "ema_21_dist",
    "ema_55_dist",
    # EMA slope (change in EMA relative to close)
    "ema_8_slope",
    "ema_21_slope",
    # VWAP deviation (requires volume)
    "vwap_dev",
    # Rolling Z-score of log returns
    "return_zscore_20",
    # Rate of change
    "roc_10",
    # OBV EMA momentum
    "obv_ema_ratio",
    # Cross-EMA signals
    "ema_8_21_cross",    # (ema_8 - ema_21) / close
    "ema_21_55_cross",   # (ema_21 - ema_55) / close
    # Momentum (close vs. N-step lagged close)
    "momentum_5",
    "momentum_20",
    # Volatility regimes
    "vol_ratio_short_long",   # std(5) / std(20) of returns
    # Raw OHLCV ratios (normalised, model-agnostic)
    "hl_range_norm",    # (high - low) / close
    "oc_range_norm",    # (open - close) / close
    # Log return
    "log_return",
    # Candle body fraction
    "body_fraction",    # |close - open| / (high - low + epsilon)
]

N_FEATURES: int = len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FeaturePipelineConfig:
    """Configuration for :class:`FeatureEngineeringPipeline`.

    Attributes:
        rsi_period: RSI look-back period (default 14).
        macd_fast: MACD fast EMA period (default 12).
        macd_slow: MACD slow EMA period (default 26).
        macd_signal_period: MACD signal EMA period (default 9).
        bb_period: Bollinger Bands rolling window (default 20).
        bb_std: Bollinger Bands standard deviation multiplier (default 2.0).
        atr_period: ATR rolling period (default 14).
        ema_periods: EMA periods to compute (default [8, 21, 55]).
        obv_ema_period: EMA period applied to OBV (default 20).
        roc_period: Rate-of-change look-back period (default 10).
        zscore_period: Rolling Z-score window for returns (default 20).
        vol_short: Short volatility window (default 5).
        vol_long: Long volatility window (default 20).
        momentum_periods: Periods for momentum features (default [5, 20]).
        ffill_then_zero: Fill NaN rows with ffill then zero (default True).
    """

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_periods: list[int] = field(default_factory=lambda: [8, 21, 55])
    obv_ema_period: int = 20
    roc_period: int = 10
    zscore_period: int = 20
    vol_short: int = 5
    vol_long: int = 20
    momentum_periods: list[int] = field(default_factory=lambda: [5, 20])
    ffill_then_zero: bool = True


# ---------------------------------------------------------------------------
# Low-level EMA helper
# ---------------------------------------------------------------------------


def _ema_series(series: pd.Series, span: int) -> pd.Series:
    """Compute an exponential moving average via pandas ewm.

    Args:
        series: Input :class:`pandas.Series`.
        span: EMA span (number of periods).

    Returns:
        EMA as a :class:`pandas.Series` aligned with *series*.
    """
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual indicator computations (all accept pd.Series, return pd.Series)
# ---------------------------------------------------------------------------


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI look-back (default 14).

    Returns:
        RSI series in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)  # neutral RSI for warm-up


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Args:
        close: Close price series.
        fast: Fast EMA span (default 12).
        slow: Slow EMA span (default 26).
        signal: Signal EMA span (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) as :class:`pandas.Series`.
    """
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_series(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper distance, lower distance, %B, bandwidth, and mid.

    Args:
        close: Close price series.
        period: Rolling window (default 20).
        num_std: Standard deviation multiplier (default 2.0).

    Returns:
        Tuple of (upper_dist, lower_dist, pct_b, bandwidth, mid).
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=1)
    upper = mid + num_std * std
    lower = mid - num_std * std

    eps = 1e-10
    upper_dist = (upper - close) / (close + eps)
    lower_dist = (close - lower) / (close + eps)
    band_range = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / band_range
    bandwidth = band_range / (mid + eps)

    return upper_dist, lower_dist, pct_b, bandwidth, mid


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range normalised by close price.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR smoothing period (default 14).

    Returns:
        ATR / close as a :class:`pandas.Series`.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    eps = 1e-10
    return atr / (close + eps)


def _compute_vwap_deviation(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling VWAP deviation: (close - vwap) / vwap.

    Args:
        close: Close price series.
        volume: Volume series.
        period: Rolling VWAP window (default 20).

    Returns:
        Deviation from rolling VWAP.
    """
    pv = close * volume
    vwap = pv.rolling(period).sum() / volume.rolling(period).sum().replace(0.0, np.nan)
    eps = 1e-10
    return (close - vwap) / (vwap + eps)


def _compute_obv_ema_ratio(close: pd.Series, volume: pd.Series, ema_period: int = 20) -> pd.Series:
    """On-Balance Volume EMA momentum: obv / ema(obv) - 1.

    Args:
        close: Close price series.
        volume: Volume series.
        ema_period: EMA smoothing period (default 20).

    Returns:
        OBV momentum ratio.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_ema = _ema_series(obv, ema_period)
    eps = 1e-10
    return obv / (obv_ema.abs() + eps) - 1.0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class FeatureEngineeringPipeline:
    """Transforms an OHLCV DataFrame into a normalised multi-indicator feature matrix.

    The pipeline accepts any DataFrame with OHLCV columns (various naming
    conventions are auto-detected).  It computes :data:`N_FEATURES` = 28
    technical indicator columns and returns them as a :class:`pandas.DataFrame`
    aligned with the original index.

    Args:
        config: :class:`FeaturePipelineConfig` with indicator parameters.
            Uses sensible defaults if not provided.

    Example::

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.transform(ohlcv_df)
        # features.shape == (len(ohlcv_df), 28)
    """

    _OPEN_COLS = ["open", "Open", "OPEN"]
    _HIGH_COLS = ["high", "High", "HIGH"]
    _LOW_COLS = ["low", "Low", "LOW"]
    _CLOSE_COLS = ["close", "Close", "CLOSE", "price", "Price", "btc_close"]
    _VOLUME_COLS = ["volume", "Volume", "VOLUME", "vol", "Vol"]

    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()
        log.info(
            "feature_engineering.init",
            rsi_period=self.config.rsi_period,
            bb_period=self.config.bb_period,
            atr_period=self.config.atr_period,
            n_features=N_FEATURES,
        )

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _find_col(self, df: pd.DataFrame, candidates: list[str], name: str) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        log.warning("feature_engineering.missing_col", col_type=name)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicator features from an OHLCV DataFrame.

        Args:
            df: Input DataFrame with OHLCV columns.  Column names are
                auto-detected from common conventions (``open/Open``,
                ``high/High``, etc.).  Volume is optional — VWAP and OBV
                features are set to zero when volume is absent.

        Returns:
            A :class:`pandas.DataFrame` with :data:`FEATURE_COLUMNS` columns,
            shape ``(len(df), N_FEATURES)``, index aligned to *df*.

        Raises:
            DataFetchError: If no close price column can be found.
        """
        close_col = self._find_col(df, self._CLOSE_COLS, "close")
        if close_col is None:
            raise DataFetchError(
                f"Cannot find close price column.  Expected one of {self._CLOSE_COLS}; "
                f"got: {list(df.columns[:10])}"
            )

        close = df[close_col].astype(float)
        high_col = self._find_col(df, self._HIGH_COLS, "high")
        low_col = self._find_col(df, self._LOW_COLS, "low")
        open_col = self._find_col(df, self._OPEN_COLS, "open")
        volume_col = self._find_col(df, self._VOLUME_COLS, "volume")

        high = df[high_col].astype(float) if high_col else close
        low = df[low_col].astype(float) if low_col else close
        open_ = df[open_col].astype(float) if open_col else close
        volume = df[volume_col].astype(float) if volume_col else pd.Series(
            np.ones(len(df), dtype=float), index=df.index
        )

        log.info(
            "feature_engineering.transform",
            rows=len(df),
            has_high=high_col is not None,
            has_low=low_col is not None,
            has_volume=volume_col is not None,
        )

        cfg = self.config
        features: dict[str, pd.Series] = {}

        # --- RSI ---
        features["rsi_14"] = _compute_rsi(close, cfg.rsi_period) / 100.0  # normalise to [0,1]

        # --- MACD ---
        macd_line, macd_signal, macd_hist = _compute_macd(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period
        )
        eps = close.mean() + 1e-10
        features["macd_line"] = macd_line / eps
        features["macd_signal"] = macd_signal / eps
        features["macd_hist"] = macd_hist / eps

        # --- Bollinger Bands ---
        bb_upper_dist, bb_lower_dist, bb_pct_b, bb_bw, _mid = _compute_bollinger(
            close, cfg.bb_period, cfg.bb_std
        )
        features["bb_upper_dist"] = bb_upper_dist
        features["bb_lower_dist"] = bb_lower_dist
        features["bb_pct_b"] = bb_pct_b
        features["bb_bandwidth"] = bb_bw

        # --- ATR ---
        features["atr_14_norm"] = _compute_atr(high, low, close, cfg.atr_period)

        # --- EMA family ---
        ema_vals: dict[int, pd.Series] = {}
        for p in cfg.ema_periods:
            ema_vals[p] = _ema_series(close, p)

        p8, p21, p55 = cfg.ema_periods[0], cfg.ema_periods[1], cfg.ema_periods[2]
        cl = close + 1e-10
        features["ema_8_dist"] = (ema_vals[p8] - close) / cl
        features["ema_21_dist"] = (ema_vals[p21] - close) / cl
        features["ema_55_dist"] = (ema_vals[p55] - close) / cl
        features["ema_8_slope"] = ema_vals[p8].diff() / cl
        features["ema_21_slope"] = ema_vals[p21].diff() / cl

        # --- VWAP deviation ---
        features["vwap_dev"] = _compute_vwap_deviation(close, volume, cfg.bb_period)

        # --- Rolling Z-score of log returns ---
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        roll_mean = log_ret.rolling(cfg.zscore_period).mean()
        roll_std = log_ret.rolling(cfg.zscore_period).std(ddof=1).replace(0.0, np.nan)
        features["return_zscore_20"] = (log_ret - roll_mean) / roll_std

        # --- Rate of change (ROC-10) ---
        prev = close.shift(cfg.roc_period).replace(0.0, np.nan)
        features["roc_10"] = (close - prev) / prev

        # --- OBV EMA ratio ---
        features["obv_ema_ratio"] = _compute_obv_ema_ratio(close, volume, cfg.obv_ema_period)

        # --- Cross-EMA signals ---
        features["ema_8_21_cross"] = (ema_vals[p8] - ema_vals[p21]) / cl
        features["ema_21_55_cross"] = (ema_vals[p21] - ema_vals[p55]) / cl

        # --- Momentum ---
        for mp in cfg.momentum_periods:
            key = f"momentum_{mp}"
            prev_m = close.shift(mp).replace(0.0, np.nan)
            features[key] = (close - prev_m) / prev_m

        # --- Volatility ratio short / long ---
        std_short = log_ret.rolling(cfg.vol_short).std(ddof=1)
        std_long = log_ret.rolling(cfg.vol_long).std(ddof=1).replace(0.0, np.nan)
        features["vol_ratio_short_long"] = std_short / std_long

        # --- OHLCV normalised ratios ---
        hl = (high - low).replace(0.0, np.nan)
        features["hl_range_norm"] = hl / cl
        features["oc_range_norm"] = (open_ - close) / cl

        # --- Log return ---
        features["log_return"] = log_ret

        # --- Candle body fraction ---
        body = (close - open_).abs()
        features["body_fraction"] = body / (hl + 1e-10)

        # Build output DataFrame in canonical column order
        out = pd.DataFrame(features, index=df.index)[FEATURE_COLUMNS]

        # Fill NaN warm-up rows
        if cfg.ffill_then_zero:
            out = out.ffill().fillna(0.0)

        # Clip extreme outliers to [-10, 10] for stability
        out = out.clip(-10.0, 10.0)

        log.info(
            "feature_engineering.done",
            output_shape=str(out.shape),
            nan_count=int(out.isna().sum().sum()),
        )
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for :meth:`transform` (stateless pipeline — no fitting required).

        Args:
            df: Input OHLCV DataFrame.

        Returns:
            Feature DataFrame as from :meth:`transform`.
        """
        return self.transform(df)


# ---------------------------------------------------------------------------
# Convenience function for single-shot use
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    config: FeaturePipelineConfig | None = None,
) -> pd.DataFrame:
    """Compute technical features from an OHLCV DataFrame in one call.

    Args:
        df: Input OHLCV DataFrame.
        config: Optional pipeline configuration.

    Returns:
        Feature DataFrame with :data:`FEATURE_COLUMNS` columns.
    """
    return FeatureEngineeringPipeline(config).transform(df)


# ---------------------------------------------------------------------------
# Feature importance / correlation report
# ---------------------------------------------------------------------------


def feature_correlation_report(features: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlation matrix for feature analysis.

    Args:
        features: Feature DataFrame as returned by :meth:`FeatureEngineeringPipeline.transform`.

    Returns:
        Symmetric correlation matrix as a :class:`pandas.DataFrame`.
    """
    return features.corr()


def top_correlated_pairs(
    features: pd.DataFrame,
    top_n: int = 10,
    threshold: float = 0.85,
) -> list[tuple[str, str, float]]:
    """Identify highly correlated feature pairs that may be redundant.

    Args:
        features: Feature DataFrame.
        top_n: Number of pairs to return.
        threshold: Minimum absolute correlation to flag.

    Returns:
        List of ``(col_a, col_b, correlation)`` tuples sorted by descending
        absolute correlation, excluding self-pairs.
    """
    corr = features.corr().abs()
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, ca in enumerate(cols):
        for cb in cols[i + 1 :]:
            r = float(corr.loc[ca, cb])
            if r >= threshold:
                pairs.append((ca, cb, r))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]

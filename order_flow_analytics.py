"""Order flow analytics module for the LARSA crypto trading system.

Provides microstructure-informed metrics derived from trade and OHLCV data,
enabling the RL agent to sense order flow imbalance, market impact, and
liquidity conditions in real time.

Features implemented:

- **VWAP** and cumulative VWAP deviation
- **Trade imbalance estimator** using the Lee-Ready tick-rule (sign of return
  used as a buy/sell classifier when tick data is absent)
- **Kyle's lambda** (price impact coefficient: Δprice / signed_volume)
- **Roll's spread estimator** (implied bid-ask spread from serial covariance)
- **Amihud illiquidity ratio** (|return| / dollar_volume)
- **Order flow toxicity** (PIN-inspired buy-sell imbalance fraction)
- **Cumulative delta** (running buy-minus-sell volume)
- **Market depth proxy** (reciprocal of ATR as a liquidity surrogate)

All computations are rolling-window–based, vectorised with NumPy/Pandas, and
designed to concatenate with :mod:`feature_engineering` output.

Example::

    from order_flow_analytics import OrderFlowAnalytics
    import pandas as pd

    df = pd.read_csv("data/BTC_1sec.csv")
    ofa = OrderFlowAnalytics(window=60)
    metrics = ofa.compute(df)
    print(metrics.columns.tolist())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exceptions import DataFetchError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column name for output DataFrame
# ---------------------------------------------------------------------------

ORDER_FLOW_COLUMNS: list[str] = [
    "vwap_dev_cum",          # close deviation from cumulative VWAP
    "tick_imbalance",        # rolling buy-tick fraction (Lee-Ready proxy)
    "kyle_lambda",           # price impact: Δprice per unit signed volume
    "roll_spread",           # Roll's implied bid-ask spread
    "amihud_illiquidity",    # |return| / dollar_volume (Amihud ratio)
    "order_toxicity",        # |buy_vol - sell_vol| / total_vol (PIN proxy)
    "cum_delta_norm",        # normalised cumulative delta (buy-sell volume)
    "depth_proxy",           # market depth surrogate (1 / ATR_norm)
    "dollar_volume_norm",    # normalised dollar volume (rolling z-score)
    "price_acceleration",    # second derivative of close (rate of change of ROC)
]

N_ORDER_FLOW_FEATURES: int = len(ORDER_FLOW_COLUMNS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OrderFlowConfig:
    """Configuration for :class:`OrderFlowAnalytics`.

    Attributes:
        window: Primary rolling window in ticks (default 60).
        vwap_window: Window for cumulative VWAP reset (0 = full cumulative).
        atr_window: ATR window for depth proxy (default 14).
        impact_window: Window for Kyle's lambda regression (default 30).
        roll_window: Window for Roll's spread estimator (default 20).
        amihud_window: Rolling window for Amihud ratio (default 30).
        zscore_window: Window for dollar-volume Z-score normalisation (default 60).
        min_volume: Minimum volume to avoid division by zero (default 1.0).
        clip_extreme: Clip each feature to this absolute value (default 5.0).
        ffill_then_zero: Fill NaN warm-up rows (default True).
    """

    window: int = 60
    vwap_window: int = 0         # 0 = cumulative from start of series
    atr_window: int = 14
    impact_window: int = 30
    roll_window: int = 20
    amihud_window: int = 30
    zscore_window: int = 60
    min_volume: float = 1.0
    clip_extreme: float = 5.0
    ffill_then_zero: bool = True


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

_CLOSE_COLS = ["close", "Close", "price", "Price", "btc_close"]
_HIGH_COLS = ["high", "High"]
_LOW_COLS = ["low", "Low"]
_OPEN_COLS = ["open", "Open"]
_VOLUME_COLS = ["volume", "Volume", "vol", "Vol"]


def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Microstructure metric helpers
# ---------------------------------------------------------------------------


def _tick_rule_sign(close: pd.Series) -> pd.Series:
    """Lee-Ready tick rule: classify each tick as buy (+1) or sell (-1).

    Returns +1 when close rises vs previous tick, -1 when it falls, and
    propagates the last non-zero direction when price is unchanged.

    Args:
        close: Close price series.

    Returns:
        Series of {-1, +1} tick directions.
    """
    diff = close.diff()
    sign = np.sign(diff)
    # propagate last non-zero sign through zero-change ticks
    sign = sign.replace(0.0, np.nan).ffill().fillna(1.0)
    return sign


def _roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll's (1984) implicit bid-ask spread estimator.

    Spread = 2 * sqrt(max(0, -Cov(Δprice_t, Δprice_{t-1}))).

    Args:
        close: Close price series.
        window: Rolling covariance window.

    Returns:
        Estimated spread series, normalised by close.
    """
    delta = close.diff()
    delta_lag = delta.shift(1)

    # rolling covariance: roll(Cov) = roll(E[X*Y]) - roll(E[X])*roll(E[Y])
    cov = (
        (delta * delta_lag)
        .rolling(window)
        .mean()
        - delta.rolling(window).mean() * delta_lag.rolling(window).mean()
    )
    spread = 2.0 * np.sqrt(np.maximum(0.0, -cov))
    eps = 1e-10
    return spread / (close + eps)


def _kyle_lambda(
    price_change: pd.Series,
    signed_volume: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Rolling Kyle's lambda: price impact per unit of signed order flow.

    Estimated as rolling OLS slope of price_change ~ signed_volume.

    Args:
        price_change: First difference of close prices.
        signed_volume: Tick-rule-classified volume (positive = buy, negative = sell).
        window: Regression window.

    Returns:
        Lambda series (units: price change per volume unit).
    """
    # Use rolling covariance / variance as OLS slope estimator
    cov_xy = (
        (price_change * signed_volume).rolling(window).mean()
        - price_change.rolling(window).mean() * signed_volume.rolling(window).mean()
    )
    var_x = signed_volume.rolling(window).var(ddof=1).replace(0.0, np.nan)
    lam = cov_xy / var_x
    # Normalise by mean absolute price change to make it scale-independent
    mean_dp = price_change.abs().rolling(window).mean().replace(0.0, np.nan)
    return lam / mean_dp


def _amihud_ratio(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio: rolling mean(|r| / dollar_volume).

    Higher values indicate lower liquidity (larger price impact per dollar traded).

    Args:
        returns: Log return series.
        dollar_volume: Dollar volume series (price * volume).
        window: Rolling window.

    Returns:
        Amihud ratio series.
    """
    eps = 1e-10
    ratio = returns.abs() / (dollar_volume + eps)
    amihud = ratio.rolling(window).mean()
    # Log-scale to compress extreme values
    return np.log1p(amihud * 1e6)


def _cumulative_delta(signed_volume: pd.Series, window: int) -> pd.Series:
    """Rolling cumulative delta (buy volume - sell volume).

    Args:
        signed_volume: Buy (+) / sell (-) classified volume.
        window: Rolling summation window.

    Returns:
        Normalised rolling cumulative delta.
    """
    cum_delta = signed_volume.rolling(window).sum()
    total_vol = signed_volume.abs().rolling(window).sum().replace(0.0, np.nan)
    return cum_delta / total_vol


def _order_toxicity(signed_volume: pd.Series, window: int) -> pd.Series:
    """PIN-inspired order toxicity: |buy_vol - sell_vol| / total_vol.

    Measures directional imbalance in the order flow.  High toxicity suggests
    informed trading (adverse selection risk).

    Args:
        signed_volume: Buy (+) / sell (-) classified volume.
        window: Rolling window.

    Returns:
        Toxicity fraction in [0, 1].
    """
    buy = signed_volume.clip(lower=0.0).rolling(window).sum()
    sell = (-signed_volume).clip(lower=0.0).rolling(window).sum()
    total = (buy + sell).replace(0.0, np.nan)
    return (buy - sell).abs() / total


def _dollar_volume_zscore(
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Z-score of dollar volume (close * volume).

    Args:
        close: Close price series.
        volume: Volume series.
        window: Z-score window.

    Returns:
        Z-score series.
    """
    dv = close * volume
    mean = dv.rolling(window).mean()
    std = dv.rolling(window).std(ddof=1).replace(0.0, np.nan)
    return (dv - mean) / std


def _price_acceleration(close: pd.Series) -> pd.Series:
    """Second derivative of close prices (rate-of-change of ROC).

    Args:
        close: Close price series.

    Returns:
        Price acceleration series, normalised by close.
    """
    eps = 1e-10
    roc = close.diff() / (close.shift(1) + eps)
    return roc.diff()


# ---------------------------------------------------------------------------
# Main analytics class
# ---------------------------------------------------------------------------


class OrderFlowAnalytics:
    """Compute order flow microstructure metrics from OHLCV data.

    Produces :data:`N_ORDER_FLOW_FEATURES` = 10 columns that complement the
    technical indicator features from :mod:`feature_engineering`.  Together
    they give the RL agent a view of both price-action and microstructure.

    Args:
        window: Primary rolling window in ticks (default 60).
        config: Optional :class:`OrderFlowConfig` for fine-grained control.

    Example::

        ofa = OrderFlowAnalytics(window=60)
        metrics = ofa.compute(ohlcv_df)
        # Concatenate with feature_engineering output:
        combined = pd.concat([tech_features, metrics], axis=1)
    """

    def __init__(
        self,
        window: int = 60,
        config: OrderFlowConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = OrderFlowConfig(window=window)

        log.info(
            "order_flow_analytics.init",
            window=self.config.window,
            n_features=N_ORDER_FLOW_FEATURES,
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all order flow metrics from an OHLCV DataFrame.

        Args:
            df: Input DataFrame with at least a close price column.
                High, low, open, and volume are used when available.

        Returns:
            A :class:`pandas.DataFrame` with :data:`ORDER_FLOW_COLUMNS` columns,
            index aligned to *df*.

        Raises:
            DataFetchError: If no close price column can be found.
        """
        close_col = _find(df, _CLOSE_COLS)
        if close_col is None:
            raise DataFetchError(
                f"Cannot find close price column in order_flow_analytics. "
                f"Expected one of {_CLOSE_COLS}; got: {list(df.columns[:10])}"
            )

        close = df[close_col].astype(float)
        high_col = _find(df, _HIGH_COLS)
        low_col = _find(df, _LOW_COLS)
        volume_col = _find(df, _VOLUME_COLS)

        high = df[high_col].astype(float) if high_col else close
        low = df[low_col].astype(float) if low_col else close
        volume = (
            df[volume_col].astype(float).clip(lower=self.config.min_volume)
            if volume_col
            else pd.Series(np.ones(len(df), dtype=float), index=df.index)
        )

        log.info(
            "order_flow_analytics.compute",
            rows=len(df),
            has_high=high_col is not None,
            has_low=low_col is not None,
            has_volume=volume_col is not None,
        )

        cfg = self.config
        w = cfg.window
        eps = 1e-10

        # --- Derived series ---
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        price_change = close.diff()
        tick_sign = _tick_rule_sign(close)
        signed_volume = tick_sign * volume
        dollar_volume = close * volume

        features: dict[str, pd.Series] = {}

        # 1. Cumulative VWAP deviation
        if cfg.vwap_window > 0:
            pv_roll = (close * volume).rolling(cfg.vwap_window).sum()
            vol_roll = volume.rolling(cfg.vwap_window).sum().replace(0.0, np.nan)
            vwap = pv_roll / vol_roll
        else:
            pv_cum = (close * volume).cumsum()
            vol_cum = volume.cumsum().replace(0.0, np.nan)
            vwap = pv_cum / vol_cum
        features["vwap_dev_cum"] = (close - vwap) / (vwap + eps)

        # 2. Tick imbalance (fraction of up-ticks in window)
        up_ticks = (tick_sign > 0).astype(float).rolling(w).sum()
        total_ticks = pd.Series(float(w), index=df.index)
        features["tick_imbalance"] = (up_ticks / total_ticks) * 2.0 - 1.0  # [-1, 1]

        # 3. Kyle's lambda
        features["kyle_lambda"] = _kyle_lambda(price_change, signed_volume, cfg.impact_window)

        # 4. Roll's spread
        features["roll_spread"] = _roll_spread(close, cfg.roll_window)

        # 5. Amihud illiquidity
        features["amihud_illiquidity"] = _amihud_ratio(log_ret, dollar_volume, cfg.amihud_window)

        # 6. Order toxicity (PIN proxy)
        features["order_toxicity"] = _order_toxicity(signed_volume, w)

        # 7. Normalised cumulative delta
        features["cum_delta_norm"] = _cumulative_delta(signed_volume, w)

        # 8. Market depth proxy: 1 / ATR_norm (higher ATR = lower depth)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / cfg.atr_window, adjust=False).mean()
        atr_norm = atr / (close + eps)
        features["depth_proxy"] = 1.0 / (atr_norm + eps)
        # Normalise depth proxy to Z-score
        d = features["depth_proxy"]
        features["depth_proxy"] = (d - d.rolling(w).mean()) / (
            d.rolling(w).std(ddof=1).replace(0.0, np.nan) + eps
        )

        # 9. Dollar volume Z-score
        features["dollar_volume_norm"] = _dollar_volume_zscore(close, volume, cfg.zscore_window)

        # 10. Price acceleration
        features["price_acceleration"] = _price_acceleration(close)

        # Assemble output
        out = pd.DataFrame(features, index=df.index)[ORDER_FLOW_COLUMNS]

        # Fill NaN warm-up rows
        if cfg.ffill_then_zero:
            out = out.ffill().fillna(0.0)

        # Clip extremes for numerical stability
        out = out.clip(-cfg.clip_extreme, cfg.clip_extreme)

        log.info(
            "order_flow_analytics.done",
            output_shape=str(out.shape),
            nan_count=int(out.isna().sum().sum()),
        )
        return out

    def compute_combined(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute order flow metrics and concatenate with technical features.

        Imports :mod:`feature_engineering` and concatenates both feature sets
        into a single DataFrame for convenience.

        Args:
            df: Input OHLCV DataFrame.

        Returns:
            Combined DataFrame with technical + order flow features.
        """
        from feature_engineering import FeatureEngineeringPipeline

        tech = FeatureEngineeringPipeline().transform(df)
        ofa = self.compute(df)
        combined = pd.concat([tech, ofa], axis=1)
        log.info(
            "order_flow_analytics.combined",
            tech_cols=len(tech.columns),
            ofa_cols=len(ofa.columns),
            total_cols=len(combined.columns),
        )
        return combined


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------


def order_flow_summary(metrics: pd.DataFrame) -> dict[str, float]:
    """Compute summary statistics for order flow metrics.

    Args:
        metrics: Output of :meth:`OrderFlowAnalytics.compute`.

    Returns:
        Dictionary with mean, std, and last value for each metric.
    """
    summary: dict[str, float] = {}
    for col in metrics.columns:
        s = metrics[col].dropna()
        if len(s) == 0:
            continue
        summary[f"{col}_mean"] = float(s.mean())
        summary[f"{col}_std"] = float(s.std())
        summary[f"{col}_last"] = float(s.iloc[-1])
    return summary

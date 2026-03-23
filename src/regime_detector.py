"""Market regime detection for FinRL DeepSeek Crypto Trading.

Classifies the current market state (trending, ranging, volatile, etc.)
using linear regression slope, R², and rolling standard deviation.
No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import math
import statistics
import unittest
from collections import Counter
from enum import Enum, auto
from typing import Dict, List


# ---------------------------------------------------------------------------
# MarketRegime
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    """Qualitative description of the current market condition."""

    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    BREAKOUT = auto()


# ---------------------------------------------------------------------------
# Simple linear regression helpers
# ---------------------------------------------------------------------------


def _linear_regression(x: List[float], y: List[float]):
    """Ordinary least-squares linear regression.

    Returns
    -------
    slope : float
    intercept : float
    r_squared : float
        Coefficient of determination R².
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0, 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R² = 1 - SS_res / SS_tot
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return slope, intercept, max(0.0, min(1.0, r_squared))


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Detect market regime from price and return series.

    Parameters
    ----------
    slope_threshold:
        Minimum absolute slope (normalised by mean price) to call a trend.
    r2_threshold:
        Minimum R² value to confirm a trend (not just noise).
    volatility_threshold:
        Annualised-equivalent rolling volatility above which we classify as
        VOLATILE.
    breakout_vol_multiplier:
        If current volatility exceeds ``volatility_threshold × multiplier``,
        classify as BREAKOUT.
    """

    def __init__(
        self,
        slope_threshold: float = 0.002,
        r2_threshold: float = 0.6,
        volatility_threshold: float = 0.03,
        breakout_vol_multiplier: float = 2.0,
    ) -> None:
        self.slope_threshold = slope_threshold
        self.r2_threshold = r2_threshold
        self.volatility_threshold = volatility_threshold
        self.breakout_vol_multiplier = breakout_vol_multiplier

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trend(
        self, prices: List[float], window: int = 20
    ) -> MarketRegime:
        """Classify the trend using OLS slope and R² over the last ``window`` bars.

        Parameters
        ----------
        prices:
            Time-ordered price series (oldest first).
        window:
            Number of most-recent bars to use.

        Returns
        -------
        MarketRegime
        """
        if len(prices) < 2:
            return MarketRegime.RANGING

        subset = prices[-window:]
        x = list(range(len(subset)))
        slope, _, r2 = _linear_regression(x, subset)

        mean_price = statistics.mean(subset)
        if mean_price == 0.0:
            return MarketRegime.RANGING

        # Normalise slope by mean price so it is scale-independent.
        norm_slope = slope / mean_price

        if r2 >= self.r2_threshold and norm_slope > self.slope_threshold:
            return MarketRegime.TRENDING_UP
        if r2 >= self.r2_threshold and norm_slope < -self.slope_threshold:
            return MarketRegime.TRENDING_DOWN
        return MarketRegime.RANGING

    # ------------------------------------------------------------------
    # Volatility detection
    # ------------------------------------------------------------------

    def detect_volatility(
        self, returns: List[float], window: int = 20
    ) -> float:
        """Compute rolling standard deviation of returns over the last ``window`` bars.

        Parameters
        ----------
        returns:
            Time-ordered return series.
        window:
            Look-back window.

        Returns
        -------
        float
            Rolling standard deviation (0.0 if fewer than 2 data points).
        """
        if len(returns) < 2:
            return 0.0
        subset = returns[-window:]
        if len(subset) < 2:
            return 0.0
        return statistics.stdev(subset)

    # ------------------------------------------------------------------
    # Combined classification
    # ------------------------------------------------------------------

    def classify_regime(self, prices: List[float]) -> MarketRegime:
        """Classify the current regime from a price series.

        Priority: BREAKOUT > VOLATILE > TRENDING_UP/DOWN > RANGING.

        Parameters
        ----------
        prices:
            Time-ordered price series.

        Returns
        -------
        MarketRegime
        """
        if len(prices) < 3:
            return MarketRegime.RANGING

        # Compute returns.
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] != 0.0
        ]

        vol = self.detect_volatility(returns)

        # Check for breakout first (very high vol).
        if vol > self.volatility_threshold * self.breakout_vol_multiplier:
            return MarketRegime.BREAKOUT

        # Then general high-volatility.
        if vol > self.volatility_threshold:
            return MarketRegime.VOLATILE

        # Otherwise check trend.
        return self.detect_trend(prices)

    # ------------------------------------------------------------------
    # Regime duration
    # ------------------------------------------------------------------

    @staticmethod
    def regime_duration(regimes: List[MarketRegime]) -> Dict[str, int]:
        """Count consecutive bars in each regime segment.

        Returns a dict mapping regime name → count of bars in the *current*
        consecutive run.  If the list is empty, returns an empty dict.

        Parameters
        ----------
        regimes:
            Time-ordered list of regime labels.

        Returns
        -------
        Dict[str, int]
            ``{"TRENDING_UP": 5, "RANGING": 3, ...}`` — bars per regime in
            the most recent consecutive run of each regime type.
        """
        if not regimes:
            return {}

        # Count the length of the most-recent consecutive run per regime.
        result: Dict[str, int] = {}
        current = regimes[-1]
        count = 0
        for r in reversed(regimes):
            if r == current:
                count += 1
            else:
                break
        result[current.name] = count

        # Also provide total counts across the whole series for completeness.
        total_counts = Counter(r.name for r in regimes)
        for name, total in total_counts.items():
            if name not in result:
                result[name] = 0  # not the current run, but track it

        return result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestLinearRegression(unittest.TestCase):
    def test_perfect_uptrend(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope, intercept, r2 = _linear_regression(x, y)
        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertAlmostEqual(intercept, 1.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_flat_line(self):
        x = [0.0, 1.0, 2.0]
        y = [5.0, 5.0, 5.0]
        slope, _, r2 = _linear_regression(x, y)
        self.assertAlmostEqual(slope, 0.0, places=6)

    def test_single_point(self):
        slope, intercept, r2 = _linear_regression([1.0], [3.0])
        self.assertAlmostEqual(slope, 0.0, places=6)
        self.assertAlmostEqual(r2, 0.0, places=6)


class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RegimeDetector(
            slope_threshold=0.002,
            r2_threshold=0.6,
            volatility_threshold=0.02,
            breakout_vol_multiplier=2.0,
        )

    # --- detect_trend ---

    def test_strong_uptrend(self):
        # Steadily rising prices → TRENDING_UP
        prices = [100.0 + i * 5 for i in range(25)]
        regime = self.detector.detect_trend(prices)
        self.assertEqual(regime, MarketRegime.TRENDING_UP)

    def test_strong_downtrend(self):
        prices = [200.0 - i * 5 for i in range(25)]
        regime = self.detector.detect_trend(prices)
        self.assertEqual(regime, MarketRegime.TRENDING_DOWN)

    def test_flat_prices_ranging(self):
        prices = [100.0] * 30
        regime = self.detector.detect_trend(prices)
        self.assertEqual(regime, MarketRegime.RANGING)

    def test_single_price_ranging(self):
        regime = self.detector.detect_trend([100.0])
        self.assertEqual(regime, MarketRegime.RANGING)

    # --- detect_volatility ---

    def test_zero_returns_zero_vol(self):
        returns = [0.0] * 20
        self.assertAlmostEqual(self.detector.detect_volatility(returns), 0.0)

    def test_vol_positive_for_varied_returns(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02] * 4
        vol = self.detector.detect_volatility(returns)
        self.assertGreater(vol, 0.0)

    def test_single_return_zero_vol(self):
        self.assertAlmostEqual(self.detector.detect_volatility([0.05]), 0.0)

    # --- classify_regime ---

    def test_classify_uptrend(self):
        prices = [100.0 + i * 5 for i in range(30)]
        self.assertEqual(self.detector.classify_regime(prices), MarketRegime.TRENDING_UP)

    def test_classify_downtrend(self):
        prices = [300.0 - i * 5 for i in range(30)]
        self.assertEqual(self.detector.classify_regime(prices), MarketRegime.TRENDING_DOWN)

    def test_classify_volatile(self):
        import random as _random
        rng = _random.Random(42)
        # Large random fluctuations
        prices = [100.0]
        for _ in range(29):
            prices.append(prices[-1] * (1 + rng.uniform(-0.1, 0.1)))
        result = self.detector.classify_regime(prices)
        self.assertIn(result, {MarketRegime.VOLATILE, MarketRegime.BREAKOUT})

    def test_classify_short_series(self):
        self.assertEqual(self.detector.classify_regime([100.0, 101.0]), MarketRegime.RANGING)

    # --- regime_duration ---

    def test_empty_regime_list(self):
        self.assertEqual(RegimeDetector.regime_duration([]), {})

    def test_single_regime(self):
        regimes = [MarketRegime.TRENDING_UP] * 5
        result = RegimeDetector.regime_duration(regimes)
        self.assertEqual(result[MarketRegime.TRENDING_UP.name], 5)

    def test_mixed_regimes_current_run(self):
        regimes = (
            [MarketRegime.RANGING] * 3
            + [MarketRegime.TRENDING_UP] * 4
        )
        result = RegimeDetector.regime_duration(regimes)
        # Most recent run is TRENDING_UP for 4 bars.
        self.assertEqual(result[MarketRegime.TRENDING_UP.name], 4)

    def test_all_regimes_tracked(self):
        regimes = [
            MarketRegime.RANGING,
            MarketRegime.TRENDING_UP,
            MarketRegime.VOLATILE,
            MarketRegime.TRENDING_UP,
        ]
        result = RegimeDetector.regime_duration(regimes)
        # All regime names should appear.
        for r in regimes:
            self.assertIn(r.name, result)


if __name__ == "__main__":
    unittest.main()

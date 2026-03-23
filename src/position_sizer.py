"""Kelly criterion and volatility-based position sizing.

Implements a suite of position-sizing methods including Kelly fraction,
fractional Kelly, volatility targeting, ATR-based sizing, and portfolio heat.
"""

from __future__ import annotations

import math
import unittest
from typing import Any


class PositionSizer:
    """Kelly criterion and volatility-based position sizing methods."""

    # ------------------------------------------------------------------
    # Kelly criterion
    # ------------------------------------------------------------------

    def kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """Full Kelly fraction: f* = p - (1-p)/ratio.

        Args:
            win_prob: Probability of a winning trade [0, 1].
            win_loss_ratio: Average win / average loss (> 0).

        Returns:
            Optimal fraction of capital to risk.  Negative values indicate
            the edge is negative (bet the other side or stay flat).
        """
        if win_loss_ratio <= 0.0:
            return 0.0
        return win_prob - (1.0 - win_prob) / win_loss_ratio

    def fractional_kelly(
        self,
        win_prob: float,
        win_loss_ratio: float,
        fraction: float = 0.5,
    ) -> float:
        """Fractional Kelly: reduce the full Kelly bet by a safety factor.

        Most practitioners use half-Kelly (fraction=0.5) to reduce drawdowns.
        """
        full = self.kelly_fraction(win_prob, win_loss_ratio)
        return fraction * full

    # ------------------------------------------------------------------
    # Volatility targeting
    # ------------------------------------------------------------------

    def volatility_targeting(
        self,
        target_vol: float,
        realized_vol: float,
        current_position: float,
    ) -> float:
        """Scale `current_position` so the portfolio hits `target_vol`.

        new_position = current_position * (target_vol / realized_vol)

        Args:
            target_vol: Desired annualised volatility (e.g. 0.10 for 10%).
            realized_vol: Recently measured annualised volatility.
            current_position: Current position size (in units, dollars, etc.).

        Returns:
            Scaled position.  Returns 0.0 if realized_vol is zero or negative.
        """
        if realized_vol <= 0.0:
            return 0.0
        return current_position * (target_vol / realized_vol)

    # ------------------------------------------------------------------
    # ATR-based sizing
    # ------------------------------------------------------------------

    def atr_position_size(
        self,
        account_size: float,
        risk_per_trade_pct: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """Size a position so that an ATR-based stop risks `risk_per_trade_pct` of account.

        stop_distance = atr * atr_multiplier
        risk_dollars  = account_size * risk_per_trade_pct
        position_size = risk_dollars / stop_distance

        Args:
            account_size: Total account equity.
            risk_per_trade_pct: Fraction of account to risk per trade (e.g. 0.01 = 1%).
            atr: Average True Range of the instrument.
            atr_multiplier: How many ATRs to place the stop away.

        Returns:
            Position size in shares/contracts.  Returns 0.0 if stop is zero.
        """
        stop_distance = atr * atr_multiplier
        if stop_distance <= 0.0:
            return 0.0
        risk_dollars = account_size * risk_per_trade_pct
        return risk_dollars / stop_distance

    # ------------------------------------------------------------------
    # Max position size
    # ------------------------------------------------------------------

    def max_position_size(
        self,
        account_size: float,
        max_risk_pct: float,
        stop_distance_pct: float,
    ) -> float:
        """Maximum position size based on account risk tolerance.

        position_size = (account_size * max_risk_pct) / stop_distance_pct

        Args:
            account_size: Total account equity.
            max_risk_pct: Maximum fraction of account to risk (e.g. 0.02 = 2%).
            stop_distance_pct: Distance from entry to stop as fraction of price (e.g. 0.05 = 5%).

        Returns:
            Maximum position size (in dollar-equivalent notional).  Returns 0.0 if stop is zero.
        """
        if stop_distance_pct <= 0.0:
            return 0.0
        return (account_size * max_risk_pct) / stop_distance_pct

    # ------------------------------------------------------------------
    # Portfolio heat
    # ------------------------------------------------------------------

    def portfolio_heat(
        self,
        positions: list[dict[str, float]],
        account_size: float,
    ) -> float:
        """Total risk as percentage of account (portfolio heat).

        Each position dict must have 'risk' key representing dollar risk
        (e.g. position_size * stop_distance).

        Args:
            positions: List of dicts with at minimum a 'risk' key.
            account_size: Total account equity.

        Returns:
            Sum of all position risks as a fraction of account_size.
            Returns 0.0 if account_size is zero.
        """
        if account_size <= 0.0:
            return 0.0
        total_risk = sum(p.get("risk", 0.0) for p in positions)
        return total_risk / account_size


# ─── TESTS ────────────────────────────────────────────────────────────────────

class TestPositionSizer(unittest.TestCase):

    def setUp(self) -> None:
        self.ps = PositionSizer()

    # kelly_fraction --------------------------------------------------------

    def test_kelly_fraction_positive_edge(self) -> None:
        # p=0.6, ratio=2.0 → f* = 0.6 - 0.4/2 = 0.4
        result = self.ps.kelly_fraction(0.6, 2.0)
        self.assertAlmostEqual(result, 0.4, places=9)

    def test_kelly_fraction_even_odds(self) -> None:
        # p=0.5, ratio=1.0 → f* = 0.5 - 0.5 = 0.0
        result = self.ps.kelly_fraction(0.5, 1.0)
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_kelly_fraction_negative_edge(self) -> None:
        # p=0.4, ratio=1.0 → f* = 0.4 - 0.6 = -0.2
        result = self.ps.kelly_fraction(0.4, 1.0)
        self.assertAlmostEqual(result, -0.2, places=9)

    def test_kelly_fraction_zero_ratio(self) -> None:
        result = self.ps.kelly_fraction(0.6, 0.0)
        self.assertEqual(result, 0.0)

    # fractional_kelly ------------------------------------------------------

    def test_fractional_kelly_half(self) -> None:
        full = self.ps.kelly_fraction(0.6, 2.0)
        half = self.ps.fractional_kelly(0.6, 2.0, 0.5)
        self.assertAlmostEqual(half, full * 0.5, places=9)

    def test_fractional_kelly_full(self) -> None:
        full = self.ps.kelly_fraction(0.6, 2.0)
        result = self.ps.fractional_kelly(0.6, 2.0, 1.0)
        self.assertAlmostEqual(result, full, places=9)

    def test_fractional_kelly_quarter(self) -> None:
        result = self.ps.fractional_kelly(0.6, 2.0, 0.25)
        self.assertAlmostEqual(result, 0.4 * 0.25, places=9)

    # volatility_targeting --------------------------------------------------

    def test_volatility_targeting_upscale(self) -> None:
        # target=0.2, realized=0.1 → scale 2x
        result = self.ps.volatility_targeting(0.2, 0.1, 100.0)
        self.assertAlmostEqual(result, 200.0, places=9)

    def test_volatility_targeting_downscale(self) -> None:
        # target=0.1, realized=0.2 → scale 0.5x
        result = self.ps.volatility_targeting(0.1, 0.2, 100.0)
        self.assertAlmostEqual(result, 50.0, places=9)

    def test_volatility_targeting_zero_realized_vol(self) -> None:
        result = self.ps.volatility_targeting(0.1, 0.0, 100.0)
        self.assertEqual(result, 0.0)

    def test_volatility_targeting_matching_vol(self) -> None:
        result = self.ps.volatility_targeting(0.15, 0.15, 50.0)
        self.assertAlmostEqual(result, 50.0, places=9)

    # atr_position_size -----------------------------------------------------

    def test_atr_position_size_basic(self) -> None:
        # account=10000, risk=1%, atr=5, multiplier=2 → stop=10, risk$=100, size=10
        result = self.ps.atr_position_size(10_000.0, 0.01, 5.0, 2.0)
        self.assertAlmostEqual(result, 10.0, places=9)

    def test_atr_position_size_zero_atr(self) -> None:
        result = self.ps.atr_position_size(10_000.0, 0.01, 0.0)
        self.assertEqual(result, 0.0)

    def test_atr_position_size_default_multiplier(self) -> None:
        # multiplier default is 2.0
        result1 = self.ps.atr_position_size(10_000.0, 0.02, 10.0)
        result2 = self.ps.atr_position_size(10_000.0, 0.02, 10.0, 2.0)
        self.assertAlmostEqual(result1, result2, places=9)

    # max_position_size -----------------------------------------------------

    def test_max_position_size_basic(self) -> None:
        # account=50000, max_risk=2%, stop=5% → size=50000*0.02/0.05=20000
        result = self.ps.max_position_size(50_000.0, 0.02, 0.05)
        self.assertAlmostEqual(result, 20_000.0, places=9)

    def test_max_position_size_zero_stop(self) -> None:
        result = self.ps.max_position_size(50_000.0, 0.02, 0.0)
        self.assertEqual(result, 0.0)

    def test_max_position_size_tight_stop(self) -> None:
        # Tight stop → larger position (more shares to risk same dollar amount).
        loose = self.ps.max_position_size(10_000.0, 0.01, 0.10)
        tight = self.ps.max_position_size(10_000.0, 0.01, 0.01)
        self.assertGreater(tight, loose)

    # portfolio_heat --------------------------------------------------------

    def test_portfolio_heat_basic(self) -> None:
        positions = [{"risk": 500.0}, {"risk": 300.0}, {"risk": 200.0}]
        heat = self.ps.portfolio_heat(positions, 10_000.0)
        self.assertAlmostEqual(heat, 0.10, places=9)  # 1000/10000

    def test_portfolio_heat_empty_positions(self) -> None:
        heat = self.ps.portfolio_heat([], 10_000.0)
        self.assertEqual(heat, 0.0)

    def test_portfolio_heat_zero_account(self) -> None:
        heat = self.ps.portfolio_heat([{"risk": 100.0}], 0.0)
        self.assertEqual(heat, 0.0)

    def test_portfolio_heat_missing_risk_key(self) -> None:
        # Positions without 'risk' key contribute 0.
        positions = [{"notional": 1000.0}, {"risk": 200.0}]
        heat = self.ps.portfolio_heat(positions, 10_000.0)
        self.assertAlmostEqual(heat, 0.02, places=9)


if __name__ == "__main__":
    unittest.main()

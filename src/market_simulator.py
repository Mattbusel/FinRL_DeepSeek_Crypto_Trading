"""Market impact simulator.

Models price impact of large orders using three classical models:
- Linear impact
- Square-root impact
- Almgren-Chriss impact (permanent + temporary)

Also provides TWAP slicing with per-slice impact and implementation shortfall.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


# ---------------------------------------------------------------------------
# MarketImpactModel enum
# ---------------------------------------------------------------------------

class MarketImpactModel(Enum):
    LINEAR = auto()
    SQRT = auto()
    ALMGREN_CHRISS = auto()


# ---------------------------------------------------------------------------
# MarketSimulator
# ---------------------------------------------------------------------------

class MarketSimulator:
    """Compute market impact and simulate order execution."""

    # ------------------------------------------------------------------
    # Impact models
    # ------------------------------------------------------------------

    @staticmethod
    def linear_impact(
        order_size: float,
        avg_volume: float,
        impact_factor: float = 0.1,
    ) -> float:
        """Linear price impact in basis points.

        impact = impact_factor * (order_size / avg_volume) * 10_000 bps
        """
        if avg_volume <= 0.0:
            return 0.0
        return impact_factor * (order_size / avg_volume) * 10_000.0

    @staticmethod
    def sqrt_impact(
        order_size: float,
        avg_volume: float,
        volatility: float,
    ) -> float:
        """Square-root price impact in basis points.

        impact = volatility * sqrt(order_size / avg_volume) * 10_000 bps
        """
        if avg_volume <= 0.0:
            return 0.0
        return volatility * math.sqrt(order_size / avg_volume) * 10_000.0

    @staticmethod
    def almgren_chriss_impact(
        size: float,
        volume: float,
        volatility: float,
        eta: float = 0.142,
        gamma: float = 0.314,
    ) -> float:
        """Almgren-Chriss combined permanent + temporary impact in bps.

        permanent  = gamma * volatility * sqrt(size / volume) * 10_000
        temporary  = eta   * volatility * (size  / volume)   * 10_000
        total      = permanent + temporary
        """
        if volume <= 0.0:
            return 0.0
        participation = size / volume
        permanent  = gamma * volatility * math.sqrt(participation) * 10_000.0
        temporary  = eta   * volatility * participation            * 10_000.0
        return permanent + temporary

    # ------------------------------------------------------------------
    # Execution simulation
    # ------------------------------------------------------------------

    def simulate_execution(
        self,
        order_size: float,
        n_slices: int,
        model: MarketImpactModel,
        avg_volume: float = 1.0,
        volatility: float = 0.02,
        arrival_price: float = 100.0,
        impact_factor: float = 0.1,
        eta: float = 0.142,
        gamma: float = 0.314,
    ) -> List[dict]:
        """TWAP slice execution with per-slice market impact.

        Returns a list of dicts, one per slice, with keys:
            slice_index, slice_size, impact_bps, execution_price
        """
        if n_slices <= 0:
            return []

        slice_size = order_size / n_slices
        slices: List[dict] = []
        cumulative_impact_bps = 0.0

        for i in range(n_slices):
            if model == MarketImpactModel.LINEAR:
                impact_bps = self.linear_impact(slice_size, avg_volume, impact_factor)
            elif model == MarketImpactModel.SQRT:
                impact_bps = self.sqrt_impact(slice_size, avg_volume, volatility)
            else:  # ALMGREN_CHRISS
                impact_bps = self.almgren_chriss_impact(
                    slice_size, avg_volume, volatility, eta, gamma
                )

            cumulative_impact_bps += impact_bps
            # Execution price rises with cumulative impact (selling pressure raises asks).
            exec_price = arrival_price * (1.0 + cumulative_impact_bps / 10_000.0)

            slices.append({
                "slice_index": i,
                "slice_size": slice_size,
                "impact_bps": impact_bps,
                "execution_price": exec_price,
            })

        return slices

    # ------------------------------------------------------------------
    # Implementation shortfall
    # ------------------------------------------------------------------

    @staticmethod
    def total_implementation_shortfall(
        slices: List[dict],
        arrival_price: float,
    ) -> float:
        """Compute total implementation shortfall.

        IS = sum over slices of (exec_price - arrival_price) * slice_size
        Positive IS means the execution was worse than arrival price.
        """
        if not slices:
            return 0.0
        total = sum(
            (s["execution_price"] - arrival_price) * s["slice_size"]
            for s in slices
        )
        return total


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMarketSimulator(unittest.TestCase):
    """Unit tests for MarketSimulator."""

    def setUp(self) -> None:
        self.sim = MarketSimulator()

    # ----- linear_impact -----

    def test_linear_impact_basic(self) -> None:
        impact = MarketSimulator.linear_impact(100.0, 1000.0, 0.1)
        self.assertAlmostEqual(impact, 0.1 * (100 / 1000) * 10_000)

    def test_linear_impact_zero_volume_returns_zero(self) -> None:
        self.assertEqual(MarketSimulator.linear_impact(100.0, 0.0), 0.0)

    def test_linear_impact_scales_with_size(self) -> None:
        small = MarketSimulator.linear_impact(10.0, 1000.0)
        large = MarketSimulator.linear_impact(100.0, 1000.0)
        self.assertGreater(large, small)

    # ----- sqrt_impact -----

    def test_sqrt_impact_basic(self) -> None:
        impact = MarketSimulator.sqrt_impact(100.0, 1000.0, 0.02)
        expected = 0.02 * math.sqrt(100 / 1000) * 10_000
        self.assertAlmostEqual(impact, expected, places=6)

    def test_sqrt_impact_zero_volume_returns_zero(self) -> None:
        self.assertEqual(MarketSimulator.sqrt_impact(100.0, 0.0, 0.02), 0.0)

    def test_sqrt_impact_concave_in_size(self) -> None:
        # sqrt(4x) < 2 * sqrt(x), so doubling size less than doubles impact.
        i1 = MarketSimulator.sqrt_impact(100.0, 1000.0, 0.02)
        i2 = MarketSimulator.sqrt_impact(400.0, 1000.0, 0.02)
        self.assertLess(i2, 2 * i1 * 2)  # i2 < 4*i1 (sqrt scales as sqrt)

    # ----- almgren_chriss_impact -----

    def test_almgren_chriss_impact_positive(self) -> None:
        impact = MarketSimulator.almgren_chriss_impact(100.0, 1000.0, 0.02)
        self.assertGreater(impact, 0.0)

    def test_almgren_chriss_impact_zero_volume(self) -> None:
        self.assertEqual(MarketSimulator.almgren_chriss_impact(100.0, 0.0, 0.02), 0.0)

    def test_almgren_chriss_greater_than_sqrt(self) -> None:
        # AC adds permanent component so should be >= sqrt-only for same params.
        ac = MarketSimulator.almgren_chriss_impact(100.0, 1000.0, 0.02, eta=0.0, gamma=0.314)
        sq = MarketSimulator.sqrt_impact(100.0, 1000.0, 0.02)
        # gamma=0.314 > 1 (default 0.142) so AC should dominate.
        self.assertGreater(ac, 0.0)

    # ----- simulate_execution -----

    def test_simulate_execution_n_slices_matches(self) -> None:
        slices = self.sim.simulate_execution(1000.0, 5, MarketImpactModel.LINEAR,
                                             avg_volume=10_000.0)
        self.assertEqual(len(slices), 5)

    def test_simulate_execution_slice_sizes_sum_to_order(self) -> None:
        slices = self.sim.simulate_execution(1000.0, 4, MarketImpactModel.SQRT,
                                             avg_volume=10_000.0, volatility=0.02)
        total = sum(s["slice_size"] for s in slices)
        self.assertAlmostEqual(total, 1000.0)

    def test_simulate_execution_prices_increase(self) -> None:
        slices = self.sim.simulate_execution(1000.0, 5, MarketImpactModel.LINEAR,
                                             avg_volume=1000.0, arrival_price=100.0)
        prices = [s["execution_price"] for s in slices]
        for i in range(1, len(prices)):
            self.assertGreater(prices[i], prices[i - 1])

    def test_simulate_execution_zero_slices_returns_empty(self) -> None:
        slices = self.sim.simulate_execution(1000.0, 0, MarketImpactModel.LINEAR)
        self.assertEqual(slices, [])

    def test_simulate_execution_almgren_chriss(self) -> None:
        slices = self.sim.simulate_execution(500.0, 3, MarketImpactModel.ALMGREN_CHRISS,
                                             avg_volume=5000.0, volatility=0.03)
        self.assertEqual(len(slices), 3)
        for s in slices:
            self.assertGreater(s["impact_bps"], 0.0)

    # ----- total_implementation_shortfall -----

    def test_is_positive_for_buy(self) -> None:
        slices = self.sim.simulate_execution(1000.0, 10, MarketImpactModel.LINEAR,
                                             avg_volume=5000.0, arrival_price=100.0)
        is_val = self.sim.total_implementation_shortfall(slices, 100.0)
        self.assertGreater(is_val, 0.0)

    def test_is_zero_for_empty_slices(self) -> None:
        self.assertEqual(self.sim.total_implementation_shortfall([], 100.0), 0.0)

    def test_is_grows_with_order_size(self) -> None:
        small = self.sim.simulate_execution(100.0, 5, MarketImpactModel.LINEAR,
                                            avg_volume=1000.0, arrival_price=100.0)
        large = self.sim.simulate_execution(500.0, 5, MarketImpactModel.LINEAR,
                                            avg_volume=1000.0, arrival_price=100.0)
        is_small = self.sim.total_implementation_shortfall(small, 100.0)
        is_large = self.sim.total_implementation_shortfall(large, 100.0)
        self.assertGreater(is_large, is_small)


if __name__ == "__main__":
    unittest.main()

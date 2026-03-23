"""Trade execution scheduling and transaction cost optimisation."""

from __future__ import annotations

import math
import unittest
from typing import Optional


# ---------------------------------------------------------------------------
# TransactionOptimizer
# ---------------------------------------------------------------------------


class TransactionOptimizer:
    """Optimise trade execution schedules to minimise transaction costs.

    Implements TWAP, VWAP, POV, and Almgren-Chriss optimal execution
    algorithms.
    """

    # -----------------------------------------------------------------------
    # TWAP
    # -----------------------------------------------------------------------

    @staticmethod
    def twap_schedule(
        total_size: float,
        n_periods: int,
        urgency: float = 0.5,
    ) -> list[float]:
        """Generate a Time-Weighted Average Price (TWAP) execution schedule.

        With urgency == 0.5 the schedule is perfectly uniform. Higher urgency
        front-loads execution; lower urgency back-loads.

        Args:
            total_size: Total order quantity to execute.
            n_periods: Number of execution periods.
            urgency: In [0, 1]. 0.5 = uniform; >0.5 = front-loaded.

        Returns:
            List of per-period quantities summing to total_size.

        Raises:
            ValueError: If arguments are out of range.
        """
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        urgency = max(0.0, min(1.0, urgency))

        if n_periods == 1:
            return [total_size]

        # Generate weights biased by urgency using a simple geometric decay
        if abs(urgency - 0.5) < 1e-9:
            weights = [1.0] * n_periods
        else:
            # decay factor: urgency > 0.5 => weights decrease (front-load)
            decay = 1.0 - 2.0 * (urgency - 0.5) * 0.5
            weights = [decay ** i for i in range(n_periods)]

        total_weight = sum(weights)
        return [total_size * w / total_weight for w in weights]

    # -----------------------------------------------------------------------
    # VWAP
    # -----------------------------------------------------------------------

    @staticmethod
    def vwap_schedule(
        total_size: float,
        volume_profile: list[float],
    ) -> list[float]:
        """Generate a Volume-Weighted Average Price (VWAP) execution schedule.

        Slices the order proportionally to the expected volume profile.

        Args:
            total_size: Total order quantity.
            volume_profile: Expected relative volume per period (will be normalised).

        Returns:
            List of per-period quantities summing to total_size.

        Raises:
            ValueError: If volume_profile is empty or all zeros.
        """
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if not volume_profile:
            raise ValueError("volume_profile must not be empty")
        total_vol = sum(volume_profile)
        if total_vol <= 0:
            raise ValueError("volume_profile must have positive total")
        return [total_size * v / total_vol for v in volume_profile]

    # -----------------------------------------------------------------------
    # POV
    # -----------------------------------------------------------------------

    @staticmethod
    def pov_schedule(
        total_size: float,
        market_volumes: list[float],
        participation_rate: float = 0.1,
    ) -> list[float]:
        """Generate a Percentage-of-Volume (POV) execution schedule.

        Executes `participation_rate` fraction of market volume each period
        until the order is filled.

        Args:
            total_size: Total order quantity.
            market_volumes: Observed or expected market volume per period.
            participation_rate: Fraction of market volume to participate in.

        Returns:
            List of per-period quantities. The schedule stops once total_size
            is consumed; remaining periods receive 0.

        Raises:
            ValueError: If inputs are out of range.
        """
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if not market_volumes:
            raise ValueError("market_volumes must not be empty")
        if not 0 < participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")

        schedule = []
        remaining = total_size
        for vol in market_volumes:
            if remaining <= 0:
                schedule.append(0.0)
                continue
            trade = min(participation_rate * vol, remaining)
            schedule.append(trade)
            remaining -= trade
        return schedule

    # -----------------------------------------------------------------------
    # Almgren-Chriss optimal execution
    # -----------------------------------------------------------------------

    @staticmethod
    def optimal_execution(
        total_size: float,
        volatility: float,
        urgency: float,
        n_periods: int,
    ) -> list[float]:
        """Almgren-Chriss optimal execution schedule.

        Minimises the expected cost (market impact + timing risk) by trading
        off between execution speed (urgency) and market impact.

        The optimal trajectory is:
        ```
        x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
        ```
        where kappa = sqrt(urgency) and T = n_periods.

        Args:
            total_size: Total shares / notional to execute.
            volatility: Asset return volatility (annualised).
            urgency: Risk-aversion parameter (higher = trade faster).
            n_periods: Number of execution periods.

        Returns:
            Per-period trade sizes summing to total_size.
        """
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if n_periods == 1:
            return [total_size]

        kappa = math.sqrt(max(urgency, 1e-6)) * (1.0 + volatility)
        T = float(n_periods)

        # Inventory at each time step using the sinh trajectory
        def inventory(t: float) -> float:
            denom = math.sinh(kappa * T)
            if denom < 1e-12:
                return total_size * (1.0 - t / T)
            return total_size * math.sinh(kappa * (T - t)) / denom

        trades = []
        for i in range(n_periods):
            x_before = inventory(float(i))
            x_after = inventory(float(i + 1))
            trades.append(max(x_before - x_after, 0.0))

        # Normalise to ensure the total is exactly total_size
        total_traded = sum(trades)
        if total_traded > 1e-12:
            trades = [t * total_size / total_traded for t in trades]
        return trades

    # -----------------------------------------------------------------------
    # Market impact cost
    # -----------------------------------------------------------------------

    @staticmethod
    def market_impact_cost(
        schedule: list[float],
        volume_profile: list[float],
        impact_factor: float = 0.1,
    ) -> float:
        """Estimate the total market impact cost of an execution schedule.

        Uses a square-root market impact model:
        ```
        cost_t = impact_factor * size_t * sqrt(size_t / volume_t)
        ```

        Args:
            schedule: Per-period trade sizes.
            volume_profile: Per-period market volume.
            impact_factor: Proportional impact coefficient.

        Returns:
            Total estimated market impact cost.
        """
        if len(schedule) != len(volume_profile):
            raise ValueError("schedule and volume_profile must have the same length")
        total_cost = 0.0
        for size, vol in zip(schedule, volume_profile):
            if vol <= 0 or size <= 0:
                continue
            participation = size / vol
            total_cost += impact_factor * size * math.sqrt(participation)
        return total_cost

    # -----------------------------------------------------------------------
    # Schedule analytics
    # -----------------------------------------------------------------------

    @staticmethod
    def schedule_analytics(schedule: list[float]) -> dict:
        """Compute summary statistics for an execution schedule.

        Args:
            schedule: Per-period trade sizes.

        Returns:
            Dict with:
            - completion_pct: Fraction of total executed in first half of periods.
            - front_loading: Ratio of first-half to second-half execution.
            - variance: Variance of per-period trade sizes.
        """
        if not schedule:
            return {"completion_pct": 0.0, "front_loading": 1.0, "variance": 0.0}

        total = sum(schedule)
        n = len(schedule)
        half = n // 2

        first_half = sum(schedule[:half])
        second_half = sum(schedule[half:])

        completion_pct = first_half / total if total > 0 else 0.0
        front_loading = first_half / second_half if second_half > 1e-12 else float("inf")

        mean = total / n
        variance = sum((s - mean) ** 2 for s in schedule) / n

        return {
            "completion_pct": completion_pct,
            "front_loading": front_loading,
            "variance": variance,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransactionOptimizer(unittest.TestCase):
    def test_twap_sums_to_total(self):
        schedule = TransactionOptimizer.twap_schedule(1000.0, 10)
        self.assertAlmostEqual(sum(schedule), 1000.0, places=6)

    def test_twap_uniform_at_neutral_urgency(self):
        schedule = TransactionOptimizer.twap_schedule(1000.0, 4, urgency=0.5)
        for s in schedule:
            self.assertAlmostEqual(s, 250.0, places=4)

    def test_twap_front_loaded_high_urgency(self):
        schedule = TransactionOptimizer.twap_schedule(1000.0, 4, urgency=0.9)
        # First period should be larger than last
        self.assertGreater(schedule[0], schedule[-1])

    def test_twap_single_period(self):
        schedule = TransactionOptimizer.twap_schedule(500.0, 1)
        self.assertEqual(schedule, [500.0])

    def test_twap_invalid_size(self):
        with self.assertRaises(ValueError):
            TransactionOptimizer.twap_schedule(-100.0, 5)

    def test_vwap_sums_to_total(self):
        profile = [0.1, 0.2, 0.4, 0.2, 0.1]
        schedule = TransactionOptimizer.vwap_schedule(1000.0, profile)
        self.assertAlmostEqual(sum(schedule), 1000.0, places=6)

    def test_vwap_proportional(self):
        profile = [1.0, 2.0, 1.0]
        schedule = TransactionOptimizer.vwap_schedule(400.0, profile)
        self.assertAlmostEqual(schedule[1] / schedule[0], 2.0, places=6)

    def test_vwap_empty_profile(self):
        with self.assertRaises(ValueError):
            TransactionOptimizer.vwap_schedule(1000.0, [])

    def test_pov_does_not_exceed_total(self):
        # With volume=200 per period * 0.5 participation * 10 periods = 1000 total
        volumes = [200.0] * 10
        schedule = TransactionOptimizer.pov_schedule(1000.0, volumes, 0.5)
        self.assertAlmostEqual(sum(schedule), 1000.0, places=6)

    def test_pov_stops_when_filled(self):
        volumes = [10000.0] * 5
        schedule = TransactionOptimizer.pov_schedule(100.0, volumes, 0.5)
        # Should fill on the first period
        self.assertAlmostEqual(schedule[0], 100.0, places=6)
        self.assertEqual(schedule[1], 0.0)

    def test_pov_invalid_participation(self):
        with self.assertRaises(ValueError):
            TransactionOptimizer.pov_schedule(1000.0, [100.0], 1.5)

    def test_optimal_execution_sums_to_total(self):
        schedule = TransactionOptimizer.optimal_execution(1000.0, 0.2, 1.0, 10)
        self.assertAlmostEqual(sum(schedule), 1000.0, places=4)

    def test_optimal_execution_single_period(self):
        schedule = TransactionOptimizer.optimal_execution(500.0, 0.1, 0.5, 1)
        self.assertEqual(schedule, [500.0])

    def test_optimal_execution_front_loads_high_urgency(self):
        schedule = TransactionOptimizer.optimal_execution(1000.0, 0.3, 5.0, 10)
        # High urgency should front-load
        self.assertGreater(schedule[0], schedule[-1])

    def test_market_impact_cost_positive(self):
        schedule = [100.0, 100.0, 100.0]
        volumes = [1000.0, 1000.0, 1000.0]
        cost = TransactionOptimizer.market_impact_cost(schedule, volumes)
        self.assertGreater(cost, 0)

    def test_market_impact_cost_length_mismatch(self):
        with self.assertRaises(ValueError):
            TransactionOptimizer.market_impact_cost([100.0], [200.0, 300.0])

    def test_schedule_analytics_keys(self):
        schedule = [100.0] * 10
        stats = TransactionOptimizer.schedule_analytics(schedule)
        self.assertIn("completion_pct", stats)
        self.assertIn("front_loading", stats)
        self.assertIn("variance", stats)

    def test_schedule_analytics_uniform_completion(self):
        schedule = [100.0] * 10
        stats = TransactionOptimizer.schedule_analytics(schedule)
        self.assertAlmostEqual(stats["completion_pct"], 0.5, places=4)
        self.assertAlmostEqual(stats["front_loading"], 1.0, places=4)
        self.assertAlmostEqual(stats["variance"], 0.0, places=6)

    def test_schedule_analytics_empty(self):
        stats = TransactionOptimizer.schedule_analytics([])
        self.assertEqual(stats["completion_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Advanced reward shaping for RL trading agents."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import List


@dataclass
class RewardComponent:
    """A named, weighted reward term."""

    name: str
    value: float
    weight: float


class RewardShaper:
    """Computes shaped rewards for RL trading agents."""

    # ------------------------------------------------------------------
    # Individual reward components
    # ------------------------------------------------------------------

    @staticmethod
    def pnl_reward(pnl: float, scale: float = 1.0) -> float:
        """Direct P&L reward, linearly scaled.

        Args:
            pnl: Realised or unrealised profit-and-loss.
            scale: Multiplicative scale factor.

        Returns:
            Scaled P&L value.
        """
        return pnl * scale

    @staticmethod
    def risk_adjusted_reward(
        pnl: float,
        volatility: float,
        target_vol: float = 0.01,
    ) -> float:
        """Sharpe-style risk-adjusted reward.

        Scales the P&L by the ratio of target volatility to realised
        volatility, effectively normalising for risk.

        Args:
            pnl: Realised P&L for the period.
            volatility: Realised volatility for the period (annualised or
                per-step — must be consistent with *target_vol*).
            target_vol: Desired normalisation volatility.

        Returns:
            Risk-adjusted reward.
        """
        denom = max(abs(volatility), 1e-6)
        return pnl / denom * target_vol

    @staticmethod
    def drawdown_penalty(
        current_drawdown: float,
        max_allowed: float = 0.10,
    ) -> float:
        """Penalise drawdown in excess of the allowed threshold.

        Args:
            current_drawdown: Current drawdown as a positive fraction
                (e.g. 0.15 = 15% down from peak).
            max_allowed: Maximum acceptable drawdown fraction.

        Returns:
            Non-positive penalty proportional to the excess drawdown.
        """
        excess = current_drawdown - max_allowed
        if excess <= 0.0:
            return 0.0
        return -excess

    @staticmethod
    def transaction_cost_penalty(
        trade_size: float,
        fee_rate: float = 0.001,
    ) -> float:
        """Negative reward equal to the transaction cost.

        Args:
            trade_size: Notional trade size (absolute value expected).
            fee_rate: Fee as a fraction of notional (e.g. 0.001 = 10 bps).

        Returns:
            Non-positive cost reward.
        """
        return -abs(trade_size) * fee_rate

    @staticmethod
    def position_size_penalty(
        position: float,
        max_position: float = 1.0,
    ) -> float:
        """Penalise position sizes that exceed the allowed maximum.

        Args:
            position: Current position size (absolute value used).
            max_position: Maximum allowed position size.

        Returns:
            Non-positive penalty proportional to the excess leverage.
        """
        excess = abs(position) - max_position
        if excess <= 0.0:
            return 0.0
        return -excess

    # ------------------------------------------------------------------
    # Composite & normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def composite_reward(components: List[RewardComponent]) -> float:
        """Compute a weighted sum of reward components.

        Args:
            components: List of :class:`RewardComponent` items.

        Returns:
            Weighted sum: sum(component.value * component.weight).
        """
        return sum(c.value * c.weight for c in components)

    @staticmethod
    def normalize(
        reward: float,
        running_mean: float,
        running_std: float,
    ) -> float:
        """Z-score normalise a reward using running statistics.

        Args:
            reward: Raw reward value.
            running_mean: Exponential moving average of rewards.
            running_std: Exponential moving standard deviation of rewards.

        Returns:
            Normalised reward; returns the raw reward if std is near zero.
        """
        if abs(running_std) < 1e-8:
            return reward
        return (reward - running_mean) / running_std


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestRewardShaper(unittest.TestCase):
    def setUp(self) -> None:
        self.shaper = RewardShaper()

    def test_pnl_reward_positive(self) -> None:
        self.assertAlmostEqual(RewardShaper.pnl_reward(100.0, scale=1.0), 100.0)

    def test_pnl_reward_scaled(self) -> None:
        self.assertAlmostEqual(RewardShaper.pnl_reward(50.0, scale=2.0), 100.0)

    def test_pnl_reward_negative(self) -> None:
        self.assertLess(RewardShaper.pnl_reward(-30.0), 0)

    def test_risk_adjusted_reward_normalised(self) -> None:
        r = RewardShaper.risk_adjusted_reward(pnl=0.02, volatility=0.02, target_vol=0.01)
        self.assertAlmostEqual(r, 0.01)

    def test_risk_adjusted_reward_low_vol_capped(self) -> None:
        # Should not divide by zero
        r = RewardShaper.risk_adjusted_reward(pnl=1.0, volatility=0.0, target_vol=0.01)
        self.assertIsNotNone(r)
        self.assertTrue(math.isfinite(r))

    def test_drawdown_penalty_no_excess(self) -> None:
        self.assertAlmostEqual(RewardShaper.drawdown_penalty(0.05, max_allowed=0.10), 0.0)

    def test_drawdown_penalty_with_excess(self) -> None:
        p = RewardShaper.drawdown_penalty(0.20, max_allowed=0.10)
        self.assertLess(p, 0)
        self.assertAlmostEqual(p, -0.10)

    def test_transaction_cost_penalty_is_negative(self) -> None:
        p = RewardShaper.transaction_cost_penalty(trade_size=1000.0, fee_rate=0.001)
        self.assertAlmostEqual(p, -1.0)

    def test_transaction_cost_penalty_zero_size(self) -> None:
        self.assertAlmostEqual(RewardShaper.transaction_cost_penalty(0.0), 0.0)

    def test_position_size_penalty_no_excess(self) -> None:
        self.assertAlmostEqual(RewardShaper.position_size_penalty(0.5, max_position=1.0), 0.0)

    def test_position_size_penalty_with_excess(self) -> None:
        p = RewardShaper.position_size_penalty(1.5, max_position=1.0)
        self.assertAlmostEqual(p, -0.5)

    def test_composite_reward_weighted_sum(self) -> None:
        components = [
            RewardComponent("pnl", value=10.0, weight=1.0),
            RewardComponent("cost", value=-2.0, weight=0.5),
        ]
        result = RewardShaper.composite_reward(components)
        self.assertAlmostEqual(result, 10.0 - 1.0)

    def test_composite_reward_empty(self) -> None:
        self.assertAlmostEqual(RewardShaper.composite_reward([]), 0.0)

    def test_normalize_standard(self) -> None:
        n = RewardShaper.normalize(10.0, running_mean=5.0, running_std=5.0)
        self.assertAlmostEqual(n, 1.0)

    def test_normalize_zero_std_returns_raw(self) -> None:
        n = RewardShaper.normalize(42.0, running_mean=10.0, running_std=0.0)
        self.assertAlmostEqual(n, 42.0)


if __name__ == "__main__":
    unittest.main()

"""RL environment for market microstructure simulation."""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OrderType(Enum):
    """Type of order an agent can submit."""

    MARKET = "market"
    LIMIT = "limit"
    CANCEL = "cancel"


@dataclass
class MarketAction:
    """An action submitted by the RL agent."""

    action_type: OrderType
    price: Optional[float]  # None for market orders and cancels
    size: float
    side: int  # 1 = buy, -1 = sell


@dataclass
class MarketState:
    """Observable state of the simulated market microstructure."""

    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    imbalance: float  # (bid_volume - ask_volume) / total_volume in [-1, 1]
    inventory: float  # agent's net position
    unrealized_pnl: float
    step: int


@dataclass
class _OrderBook:
    """Minimal internal order book for the microstructure simulation."""

    bid_price: float
    ask_price: float
    bid_volume: float = 100.0
    ask_volume: float = 100.0

    def spread(self) -> float:
        return self.ask_price - self.bid_price

    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    def imbalance(self) -> float:
        total = self.bid_volume + self.ask_volume
        if total < 1e-9:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total


class MarketMicrostructureEnv:
    """Gym-style RL environment for market microstructure simulation.

    The environment simulates a simplified limit order book where the agent
    places market or limit orders and earns rewards based on P&L.
    """

    MAX_STEPS = 500
    INITIAL_SPREAD = 0.10  # typical bid-ask spread
    INVENTORY_LIMIT = 10.0
    BASE_VOLATILITY = 0.05

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._book: _OrderBook = _OrderBook(bid_price=99.95, ask_price=100.05)
        self._step = 0
        self._entry_price: float = 0.0
        self._inventory: float = 0.0
        self._cash: float = 0.0
        self._initial_mid: float = 100.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> MarketState:
        """Initialise the environment with a random mid price."""
        mid = 100.0 + self._rng.uniform(-5.0, 5.0)
        half_spread = self.INITIAL_SPREAD / 2.0
        self._book = _OrderBook(
            bid_price=mid - half_spread,
            ask_price=mid + half_spread,
            bid_volume=100.0 + self._rng.uniform(-20.0, 20.0),
            ask_volume=100.0 + self._rng.uniform(-20.0, 20.0),
        )
        self._step = 0
        self._inventory = 0.0
        self._cash = 0.0
        self._initial_mid = mid
        self._entry_price = mid
        self._done = False
        return self._get_state()

    def step(
        self, action: MarketAction
    ) -> Tuple[MarketState, float, bool, Dict]:
        """Apply an action and advance the environment by one step.

        Args:
            action: The agent's chosen action.

        Returns:
            (next_state, reward, done, info) tuple.
        """
        if self._done:
            raise RuntimeError("Environment is done — call reset() first.")

        prev_state = self._get_state()
        self._update_orderbook(action)
        self._evolve_price()
        self._step += 1

        reward = self._compute_reward(prev_state, action)
        done = (
            self._step >= self.MAX_STEPS
            or abs(self._inventory) > self.INVENTORY_LIMIT * 2
        )
        self._done = done
        next_state = self._get_state()
        info = {
            "step": self._step,
            "inventory": self._inventory,
            "mid_price": self._book.mid(),
        }
        return next_state, reward, done, info

    def render(self) -> str:
        """Return an ASCII order book display."""
        mid = self._book.mid()
        lines = [
            "┌──────────────────────────────────┐",
            f"│  ASK: {self._book.ask_price:>8.4f}  vol={self._book.ask_volume:>6.0f}  │",
            f"│  MID: {mid:>8.4f}                   │",
            f"│  BID: {self._book.bid_price:>8.4f}  vol={self._book.bid_volume:>6.0f}  │",
            f"│  Inventory: {self._inventory:>+8.2f}               │",
            f"│  Step:      {self._step:>8d}               │",
            "└──────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def observation_vector(self, state: MarketState) -> List[float]:
        """Convert a state to a normalised feature vector.

        Returns:
            8-element list of normalised features.
        """
        return [
            state.mid_price / self._initial_mid - 1.0,
            state.spread / self.INITIAL_SPREAD - 1.0,
            state.imbalance,
            state.inventory / self.INVENTORY_LIMIT,
            state.unrealized_pnl / max(abs(self._initial_mid), 1.0),
            state.step / self.MAX_STEPS,
            (state.best_bid / self._initial_mid) - 1.0,
            (state.best_ask / self._initial_mid) - 1.0,
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> MarketState:
        mid = self._book.mid()
        unrealized = self._inventory * (mid - self._entry_price)
        return MarketState(
            best_bid=self._book.bid_price,
            best_ask=self._book.ask_price,
            mid_price=mid,
            spread=self._book.spread(),
            imbalance=self._book.imbalance(),
            inventory=self._inventory,
            unrealized_pnl=unrealized,
            step=self._step,
        )

    def _update_orderbook(self, action: MarketAction) -> None:
        """Apply the agent's order to the internal book state."""
        if action.action_type == OrderType.CANCEL:
            return

        size = abs(action.size) * action.side

        if action.action_type == OrderType.MARKET:
            fill_price = (
                self._book.ask_price if action.side == 1 else self._book.bid_price
            )
            if self._inventory == 0.0:
                self._entry_price = fill_price
            else:
                # Update average entry price
                total = self._inventory + size
                if abs(total) > 1e-9:
                    self._entry_price = (
                        self._entry_price * self._inventory + fill_price * size
                    ) / total
            self._inventory += size
            self._cash -= fill_price * size

        elif action.action_type == OrderType.LIMIT:
            if action.price is None:
                return
            # Accept limit order if price is within the current spread.
            if action.side == 1 and action.price >= self._book.ask_price:
                self._inventory += size
                self._cash -= action.price * size
            elif action.side == -1 and action.price <= self._book.bid_price:
                self._inventory += size
                self._cash -= action.price * size

        # Keep volumes from going negative
        self._book.bid_volume = max(10.0, self._book.bid_volume)
        self._book.ask_volume = max(10.0, self._book.ask_volume)

    def _evolve_price(self) -> None:
        """Stochastically evolve the order book prices."""
        shock = self._rng.gauss(0.0, self.BASE_VOLATILITY)
        half_spread = self.INITIAL_SPREAD / 2.0
        mid = self._book.mid() + shock
        self._book.bid_price = mid - half_spread
        self._book.ask_price = mid + half_spread
        # Randomly evolve volumes
        self._book.bid_volume = max(10.0, self._book.bid_volume + self._rng.uniform(-5, 5))
        self._book.ask_volume = max(10.0, self._book.ask_volume + self._rng.uniform(-5, 5))

    def _compute_reward(self, prev_state: MarketState, action: MarketAction) -> float:
        """Compute step reward based on P&L change and action costs."""
        mid = self._book.mid()
        curr_unrealized = self._inventory * (mid - self._entry_price)
        delta_pnl = curr_unrealized - prev_state.unrealized_pnl

        # Transaction cost
        cost = -abs(action.size) * self._book.spread() * 0.5

        # Inventory penalty
        inv_penalty = -0.001 * abs(self._inventory)

        return delta_pnl + cost + inv_penalty


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestMarketMicrostructureEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MarketMicrostructureEnv(seed=42)

    def test_reset_returns_market_state(self) -> None:
        state = self.env.reset()
        self.assertIsInstance(state, MarketState)

    def test_reset_zero_inventory(self) -> None:
        state = self.env.reset()
        self.assertAlmostEqual(state.inventory, 0.0)

    def test_reset_positive_spread(self) -> None:
        state = self.env.reset()
        self.assertGreater(state.spread, 0)

    def test_reset_mid_price_near_100(self) -> None:
        state = self.env.reset()
        self.assertGreater(state.mid_price, 90.0)
        self.assertLess(state.mid_price, 110.0)

    def test_step_returns_tuple(self) -> None:
        self.env.reset()
        action = MarketAction(action_type=OrderType.MARKET, price=None, size=1.0, side=1)
        result = self.env.step(action)
        self.assertEqual(len(result), 4)

    def test_step_increments_step(self) -> None:
        self.env.reset()
        action = MarketAction(action_type=OrderType.CANCEL, price=None, size=0.0, side=1)
        state, _, _, _ = self.env.step(action)
        self.assertEqual(state.step, 1)

    def test_step_done_after_max_steps(self) -> None:
        self.env.reset()
        action = MarketAction(action_type=OrderType.CANCEL, price=None, size=0.0, side=1)
        done = False
        for _ in range(self.env.MAX_STEPS):
            _, _, done, _ = self.env.step(action)
        self.assertTrue(done)

    def test_market_buy_increases_inventory(self) -> None:
        self.env.reset()
        action = MarketAction(action_type=OrderType.MARKET, price=None, size=2.0, side=1)
        state, _, _, _ = self.env.step(action)
        self.assertGreater(state.inventory, 0)

    def test_market_sell_decreases_inventory(self) -> None:
        self.env.reset()
        buy = MarketAction(action_type=OrderType.MARKET, price=None, size=3.0, side=1)
        self.env.step(buy)
        sell = MarketAction(action_type=OrderType.MARKET, price=None, size=1.0, side=-1)
        state, _, _, _ = self.env.step(sell)
        # Net inventory should be 2
        self.assertAlmostEqual(state.inventory, 2.0, places=5)

    def test_render_returns_string(self) -> None:
        self.env.reset()
        display = self.env.render()
        self.assertIsInstance(display, str)
        self.assertIn("BID", display)
        self.assertIn("ASK", display)

    def test_observation_vector_length(self) -> None:
        state = self.env.reset()
        obs = self.env.observation_vector(state)
        self.assertEqual(len(obs), 8)

    def test_observation_vector_finite(self) -> None:
        import math
        state = self.env.reset()
        obs = self.env.observation_vector(state)
        for v in obs:
            self.assertTrue(math.isfinite(v))

    def test_reward_is_finite(self) -> None:
        import math
        self.env.reset()
        action = MarketAction(action_type=OrderType.MARKET, price=None, size=1.0, side=1)
        _, reward, _, _ = self.env.step(action)
        self.assertTrue(math.isfinite(reward))

    def test_imbalance_in_minus_one_to_one(self) -> None:
        state = self.env.reset()
        self.assertGreaterEqual(state.imbalance, -1.0)
        self.assertLessEqual(state.imbalance, 1.0)


if __name__ == "__main__":
    unittest.main()

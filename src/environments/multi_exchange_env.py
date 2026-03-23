"""Multi-exchange trading environment (gym-style)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


class MultiExchangeEnv:
    """Simulated multi-exchange, multi-asset trading environment."""

    def __init__(
        self,
        exchanges: List[str],
        assets: List[str],
        initial_capital: float = 100_000.0,
        seed: int = 42,
    ) -> None:
        self.exchanges = exchanges
        self.assets = assets
        self.initial_capital = initial_capital
        self._seed = seed
        self._lcg_state = seed

        # Internal state
        self.cash: float = initial_capital
        self.portfolio: Dict[str, Dict[str, float]] = {}
        self.prices: Dict[str, Dict[str, float]] = {}
        self._step_count = 0
        self._max_steps = 1000

        self.reset()

    # ------------------------------------------------------------------
    # LCG pseudo-random source
    # ------------------------------------------------------------------

    def _lcg_next_float(self) -> float:
        """Return a uniform float in [0, 1) via LCG."""
        self._lcg_state = (
            self._lcg_state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
        ) & 0xFFFFFFFFFFFFFFFF
        return (self._lcg_state >> 11) / (2**53)

    def _lcg_normal(self) -> float:
        """Box-Muller transform using LCG for a standard normal sample."""
        u1 = max(self._lcg_next_float(), 1e-15)
        u2 = self._lcg_next_float()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment to the initial state."""
        self._lcg_state = self._seed
        self._step_count = 0
        self.cash = self.initial_capital

        self.prices = {
            exchange: {asset: 100.0 for asset in self.assets}
            for exchange in self.exchanges
        }
        self.portfolio = {
            exchange: {asset: 0.0 for asset in self.assets}
            for exchange in self.exchanges
        }
        return self._observation()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[str, Dict[str, float]]
    ) -> Tuple[dict, float, bool, dict]:
        """
        Apply actions = {exchange: {asset: quantity}} and advance one step.
        Positive quantity = buy; negative = sell.
        Returns (obs, reward, done, info).
        """
        prev_value = self.portfolio_value()

        fee_rate = 0.001  # 0.1% fee
        slippage_rate = 0.0005  # 0.05% slippage

        for exchange in self.exchanges:
            ex_actions = actions.get(exchange, {})
            for asset in self.assets:
                qty = ex_actions.get(asset, 0.0)
                if qty == 0.0:
                    continue
                price = self.prices[exchange][asset]
                direction = 1.0 if qty > 0 else -1.0
                fill_price = price * (1.0 + direction * slippage_rate)
                cost = qty * fill_price
                fee = abs(cost) * fee_rate

                # Cash impact
                self.cash -= cost + fee
                # Portfolio update
                self.portfolio[exchange][asset] += qty

        # Advance prices
        for exchange in self.exchanges:
            for asset in self.assets:
                self.prices[exchange][asset] = self._simulate_price(exchange, asset)

        self._step_count += 1
        done = self._step_count >= self._max_steps

        reward = self._reward(prev_value)
        obs = self._observation()
        info = {"step": self._step_count, "portfolio_value": self.portfolio_value()}
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Price simulation (GBM)
    # ------------------------------------------------------------------

    def _simulate_price(self, exchange: str, asset: str) -> float:
        """GBM step: price *= exp(mu*dt + sigma*sqrt(dt)*Z)."""
        mu = 0.0001
        sigma = 0.02
        dt = 1.0 / 252.0
        z = self._lcg_normal()
        current = self.prices[exchange][asset]
        return current * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> dict:
        """Return {exchange: {asset: {price, portfolio_qty, cash}}}."""
        obs: dict = {}
        for exchange in self.exchanges:
            obs[exchange] = {}
            for asset in self.assets:
                obs[exchange][asset] = {
                    "price": self.prices[exchange][asset],
                    "portfolio_qty": self.portfolio[exchange][asset],
                    "cash": self.cash,
                }
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward(self, prev_value: float = 0.0) -> float:
        """Reward = change in total portfolio value."""
        return self.portfolio_value() - prev_value

    # ------------------------------------------------------------------
    # Portfolio value
    # ------------------------------------------------------------------

    def portfolio_value(self) -> float:
        """Total portfolio value (cash + holdings)."""
        total = self.cash
        for exchange in self.exchanges:
            for asset in self.assets:
                qty = self.portfolio[exchange][asset]
                price = self.prices[exchange][asset]
                total += qty * price
        return total

    # ------------------------------------------------------------------
    # Arbitrage detection
    # ------------------------------------------------------------------

    def arbitrage_opportunities(self) -> List[dict]:
        """
        Return asset/exchange pairs where the price differs by > 0.5%
        compared to the average across exchanges.
        """
        opportunities = []
        for asset in self.assets:
            prices_by_exchange = {
                ex: self.prices[ex][asset] for ex in self.exchanges
            }
            avg_price = sum(prices_by_exchange.values()) / len(self.exchanges)
            if avg_price == 0:
                continue
            for exchange, price in prices_by_exchange.items():
                deviation = abs(price - avg_price) / avg_price
                if deviation > 0.005:
                    opportunities.append(
                        {
                            "asset": asset,
                            "exchange": exchange,
                            "price": price,
                            "avg_price": avg_price,
                            "deviation_pct": deviation * 100,
                        }
                    )
        return opportunities

"""
Multi-asset portfolio management with rebalancing.

Manages a portfolio of assets with target weights and provides
rebalancing logic, performance metrics, and execution simulation.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Asset:
    symbol: str
    price: float
    quantity: float
    target_weight: float


@dataclass
class PortfolioState:
    assets: List[Asset]
    total_value: float
    timestamp: float
    cash: float


class MultiAssetManager:
    """Manages a multi-asset portfolio with target weights and rebalancing."""

    def __init__(self, initial_capital: float, rebalance_threshold: float = 0.05):
        self.initial_capital = initial_capital
        self.rebalance_threshold = rebalance_threshold
        self.cash = initial_capital
        self._assets: Dict[str, Asset] = {}
        self._timestamp: float = 0.0

    def add_asset(self, symbol: str, target_weight: float):
        """Add a new asset with a target portfolio weight."""
        self._assets[symbol] = Asset(
            symbol=symbol,
            price=0.0,
            quantity=0.0,
            target_weight=target_weight,
        )

    def update_prices(self, prices: Dict[str, float], timestamp: float):
        """Update asset prices from a price dict."""
        self._timestamp = timestamp
        for symbol, price in prices.items():
            if symbol in self._assets:
                self._assets[symbol].price = price

    def _total_value(self) -> float:
        """Total portfolio value: cash + sum(price * quantity)."""
        asset_value = sum(a.price * a.quantity for a in self._assets.values())
        return self.cash + asset_value

    def current_weights(self) -> Dict[str, float]:
        """Current weights: market_value / total_value."""
        total = self._total_value()
        if total <= 0:
            return {s: 0.0 for s in self._assets}
        return {
            symbol: (a.price * a.quantity) / total
            for symbol, a in self._assets.items()
        }

    def weight_deviations(self) -> Dict[str, float]:
        """Deviation of current weights from target weights."""
        weights = self.current_weights()
        return {
            symbol: weights.get(symbol, 0.0) - a.target_weight
            for symbol, a in self._assets.items()
        }

    def needs_rebalance(self) -> bool:
        """True if any |deviation| > threshold."""
        deviations = self.weight_deviations()
        return any(abs(d) > self.rebalance_threshold for d in deviations.values())

    def rebalance_orders(self) -> List[Dict]:
        """
        Compute orders to restore target weights.

        Returns list of {symbol, action, quantity, value}.
        """
        total = self._total_value()
        orders = []
        for symbol, asset in self._assets.items():
            if asset.price <= 0:
                continue
            target_value = asset.target_weight * total
            current_value = asset.price * asset.quantity
            delta_value = target_value - current_value
            if abs(delta_value) < 0.01:
                continue
            qty = abs(delta_value) / asset.price
            action = "buy" if delta_value > 0 else "sell"
            orders.append({
                "symbol": symbol,
                "action": action,
                "quantity": qty,
                "value": abs(delta_value),
            })
        return orders

    def execute_rebalance(self, prices: Dict[str, float]):
        """Simulate order execution at current prices."""
        self.update_prices(prices, self._timestamp)
        orders = self.rebalance_orders()
        for order in orders:
            symbol = order["symbol"]
            qty = order["quantity"]
            value = order["value"]
            if order["action"] == "buy":
                if self.cash >= value:
                    self._assets[symbol].quantity += qty
                    self.cash -= value
            elif order["action"] == "sell":
                if self._assets[symbol].quantity >= qty:
                    self._assets[symbol].quantity -= qty
                    self.cash += value

    def portfolio_return(self, initial_value: float) -> float:
        """Simple return: (current - initial) / initial."""
        current = self._total_value()
        if initial_value <= 0:
            return 0.0
        return (current - initial_value) / initial_value

    def portfolio_volatility(self, return_history: List[float]) -> float:
        """Annualized standard deviation of returns (assuming daily returns)."""
        n = len(return_history)
        if n < 2:
            return 0.0
        mean = sum(return_history) / n
        variance = sum((r - mean) ** 2 for r in return_history) / (n - 1)
        return math.sqrt(variance) * math.sqrt(252)

    def sharpe_ratio(self, returns: List[float], risk_free: float = 0.02) -> float:
        """Annualized Sharpe ratio."""
        if not returns:
            return 0.0
        n = len(returns)
        mean_daily = sum(returns) / n
        mean_annual = mean_daily * 252
        vol = self.portfolio_volatility(returns)
        if vol < 1e-12:
            return 0.0
        return (mean_annual - risk_free) / vol

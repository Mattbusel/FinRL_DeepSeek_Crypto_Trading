"""Portfolio Rebalancer for LARSA.

Computes drift-based rebalancing trades and executes them via PaperTrader,
tracking tracking error before and after rebalancing.

Tracking error formula::

    tracking_error = sqrt(sum((w_actual - w_target)^2))

Example::

    from src.rebalancer import Rebalancer, RebalanceConfig

    config = RebalanceConfig(
        target_weights={"BTCUSDT": 0.5, "ETHUSDT": 0.3, "BNBUSDT": 0.2},
        threshold_pct=0.05,
        rebalance_frequency="daily",
        max_trade_size_usd=10_000.0,
    )
    rebalancer = Rebalancer(config)
    trades = rebalancer.compute_trades(portfolio_snapshot, prices)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RebalanceConfig:
    """Configuration for the portfolio rebalancer.

    Attributes:
        target_weights: Target portfolio weights keyed by symbol.
            Must sum to approximately 1.0.
        threshold_pct: Minimum absolute drift (e.g. 0.05 = 5 pp) before a
            trade is triggered.
        rebalance_frequency: Human-readable frequency label ('daily',
            'weekly', etc.).  Used for logging only; scheduling is external.
        max_trade_size_usd: Upper bound on a single trade notional in USD.
    """

    target_weights: Dict[str, float]
    threshold_pct: float = 0.05
    rebalance_frequency: str = "daily"
    max_trade_size_usd: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.target_weights:
            raise ValueError("target_weights must not be empty")
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"target_weights must sum to 1.0, got {total:.6f}"
            )
        if self.threshold_pct < 0:
            raise ValueError("threshold_pct must be non-negative")
        if self.max_trade_size_usd <= 0:
            raise ValueError("max_trade_size_usd must be positive")


@dataclass
class RebalanceTrade:
    """A single rebalancing trade produced by the Rebalancer.

    Attributes:
        symbol: Asset symbol.
        current_weight: Actual portfolio weight before rebalancing.
        target_weight: Target weight from config.
        trade_direction: ``"buy"`` or ``"sell"``.
        trade_usd: Notional USD value of the required trade (before cap).
        estimated_cost: Estimated transaction cost in USD (0.1 % of notional).
    """

    symbol: str
    current_weight: float
    target_weight: float
    trade_direction: str  # "buy" or "sell"
    trade_usd: float
    estimated_cost: float


@dataclass
class RebalanceResult:
    """Summary of a completed rebalancing operation.

    Attributes:
        trades_executed: List of trades that were sent to PaperTrader.
        total_cost_usd: Sum of estimated transaction costs.
        tracking_error_before: Tracking error before rebalancing.
        tracking_error_after: Tracking error after rebalancing (post-trade
            weights are approximated from executed trade sizes).
    """

    trades_executed: List[RebalanceTrade]
    total_cost_usd: float
    tracking_error_before: float
    tracking_error_after: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMISSION_RATE = 0.001  # 0.1 %


def _tracking_error(
    actual: Dict[str, float], target: Dict[str, float]
) -> float:
    """Compute tracking error as the Euclidean distance between weight vectors.

    Assets present in *target* but absent in *actual* are treated as having
    zero actual weight.

    Args:
        actual: Actual portfolio weights keyed by symbol.
        target: Target portfolio weights keyed by symbol.

    Returns:
        Non-negative tracking error value.
    """
    sq_sum = 0.0
    for sym, w_t in target.items():
        w_a = actual.get(sym, 0.0)
        sq_sum += (w_a - w_t) ** 2
    # Also penalise assets held but not in target
    for sym, w_a in actual.items():
        if sym not in target:
            sq_sum += w_a ** 2
    return math.sqrt(sq_sum)


def _compute_current_weights(
    portfolio_snapshot: Dict[str, float],
    prices: Dict[str, float],
) -> tuple[Dict[str, float], float]:
    """Convert a holdings snapshot to portfolio weights.

    Args:
        portfolio_snapshot: {symbol: quantity} held.
        prices: {symbol: price_usd} current prices.

    Returns:
        Tuple of (weights_dict, total_portfolio_usd).
    """
    values: Dict[str, float] = {}
    for sym, qty in portfolio_snapshot.items():
        price = prices.get(sym, 0.0)
        values[sym] = qty * price

    total = sum(values.values())
    if total <= 0:
        return {sym: 0.0 for sym in portfolio_snapshot}, 0.0

    weights = {sym: val / total for sym, val in values.items()}
    return weights, total


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------


class Rebalancer:
    """Drift-based portfolio rebalancer.

    Args:
        config: :class:`RebalanceConfig` instance defining target weights and
            trade parameters.
    """

    def __init__(self, config: RebalanceConfig) -> None:
        self._config = config

    @property
    def config(self) -> RebalanceConfig:
        """Read-only access to the rebalance configuration."""
        return self._config

    def compute_trades(
        self,
        portfolio_snapshot: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[RebalanceTrade]:
        """Find drifted positions and compute the required rebalancing trades.

        Only positions whose drift exceeds ``config.threshold_pct`` trigger a
        trade.  Each trade notional is capped at ``config.max_trade_size_usd``.

        Args:
            portfolio_snapshot: Current holdings as ``{symbol: quantity}``.
            prices: Current prices as ``{symbol: price_usd}``.

        Returns:
            List of :class:`RebalanceTrade` objects, one per symbol that
            exceeds the drift threshold.  The list is sorted by absolute drift
            descending (largest drift first).
        """
        current_weights, total_portfolio_usd = _compute_current_weights(
            portfolio_snapshot, prices
        )

        if total_portfolio_usd <= 0:
            return []

        trades: List[RebalanceTrade] = []

        all_symbols = set(self._config.target_weights) | set(current_weights)

        for sym in all_symbols:
            w_actual = current_weights.get(sym, 0.0)
            w_target = self._config.target_weights.get(sym, 0.0)
            drift = w_actual - w_target

            if abs(drift) <= self._config.threshold_pct:
                continue

            # Notional trade: bring actual weight to target weight
            raw_trade_usd = abs(drift) * total_portfolio_usd
            trade_usd = min(raw_trade_usd, self._config.max_trade_size_usd)
            direction = "sell" if drift > 0 else "buy"
            cost = trade_usd * _COMMISSION_RATE

            trades.append(
                RebalanceTrade(
                    symbol=sym,
                    current_weight=w_actual,
                    target_weight=w_target,
                    trade_direction=direction,
                    trade_usd=trade_usd,
                    estimated_cost=cost,
                )
            )

        trades.sort(key=lambda t: abs(t.current_weight - t.target_weight), reverse=True)
        return trades

    def execute(
        self,
        paper_trader: object,
        prices: Dict[str, float],
    ) -> RebalanceResult:
        """Execute rebalancing trades via *paper_trader*.

        Reads the current positions from ``paper_trader.positions``, computes
        the required trades, and sends each trade to
        ``paper_trader.buy`` / ``paper_trader.sell``.  The post-trade tracking
        error is estimated from the applied trade sizes.

        Args:
            paper_trader: An instance of :class:`~src.paper_trader.PaperTrader`
                (or any object that exposes the same interface).
            prices: Current market prices ``{symbol: price_usd}``.

        Returns:
            :class:`RebalanceResult` summarising what was traded.
        """
        # Build snapshot from PaperTrader positions
        snapshot: Dict[str, float] = {}
        positions = getattr(paper_trader, "positions", {})
        for sym, pos in positions.items():
            qty = getattr(pos, "quantity", 0.0)
            snapshot[sym] = qty

        current_weights, total_usd = _compute_current_weights(snapshot, prices)
        te_before = _tracking_error(current_weights, self._config.target_weights)

        trades = self.compute_trades(snapshot, prices)

        for trade in trades:
            price = prices.get(trade.symbol, 0.0)
            if price <= 0:
                continue
            qty = trade.trade_usd / price
            if trade.trade_direction == "buy":
                _call_if_exists(paper_trader, "buy", trade.symbol, price, qty)
            else:
                _call_if_exists(paper_trader, "sell", trade.symbol, price, qty)

        # Estimate post-trade weights
        post_snapshot = dict(snapshot)
        for trade in trades:
            price = prices.get(trade.symbol, 0.0)
            if price <= 0:
                continue
            qty_change = trade.trade_usd / price
            current_qty = post_snapshot.get(trade.symbol, 0.0)
            if trade.trade_direction == "buy":
                post_snapshot[trade.symbol] = current_qty + qty_change
            else:
                post_snapshot[trade.symbol] = max(0.0, current_qty - qty_change)

        post_weights, _ = _compute_current_weights(post_snapshot, prices)
        te_after = _tracking_error(post_weights, self._config.target_weights)

        total_cost = sum(t.estimated_cost for t in trades)

        return RebalanceResult(
            trades_executed=trades,
            total_cost_usd=total_cost,
            tracking_error_before=te_before,
            tracking_error_after=te_after,
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _call_if_exists(obj: object, method: str, *args: object) -> None:
    """Call ``obj.method(*args)`` if the method exists, else no-op."""
    fn = getattr(obj, method, None)
    if callable(fn):
        fn(*args)

"""Full backtesting framework for crypto trading strategies."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


# ---------------------------------------------------------------------------
# Config and trade data classes
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    initial_capital: float
    commission_bps: float       # commission in basis points
    slippage_bps: float         # slippage in basis points
    start_date: str             # ISO date string
    end_date: str               # ISO date string
    symbol: str


@dataclass
class BacktestTrade:
    timestamp: int              # step index
    symbol: str
    side: str                   # "BUY" or "SELL"
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float


@dataclass
class BacktestMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration_hours: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Vectorised backtesting engine.

    Signals: 1 = buy, -1 = sell, 0 = hold.
    Positions are binary: either flat (0) or long (1).
    """

    def run(
        self,
        prices: List[float],
        signals: List[int],
        config: BacktestConfig,
    ) -> BacktestMetrics:
        """Execute a backtest and return performance metrics."""
        assert len(prices) == len(signals), "prices and signals must have same length"
        n = len(prices)
        trades: List[BacktestTrade] = []
        capital = config.initial_capital
        position = 0.0          # quantity held
        entry_price = 0.0
        entry_step = 0
        gross_profit = 0.0
        gross_loss = 0.0
        winning_trades = 0
        trade_durations: List[float] = []

        for i in range(n):
            price = prices[i]
            sig = signals[i]
            bps_factor = 1e-4

            if sig == 1 and position == 0.0:
                # Enter long
                slip = price * config.slippage_bps * bps_factor
                exec_price = price + slip
                comm = exec_price * config.commission_bps * bps_factor
                qty = (capital - comm) / exec_price if exec_price > 0 else 0.0
                if qty > 0:
                    cost = qty * exec_price + comm
                    capital -= cost
                    position = qty
                    entry_price = exec_price
                    entry_step = i
                    trades.append(BacktestTrade(
                        timestamp=i, symbol=config.symbol, side="BUY",
                        quantity=qty, price=exec_price,
                        commission=comm, slippage=slip, pnl=0.0,
                    ))

            elif sig == -1 and position > 0.0:
                # Exit long
                slip = price * config.slippage_bps * bps_factor
                exec_price = price - slip
                comm = position * exec_price * config.commission_bps * bps_factor
                proceeds = position * exec_price - comm
                pnl = proceeds - (position * entry_price)
                capital += proceeds
                if pnl > 0:
                    gross_profit += pnl
                    winning_trades += 1
                else:
                    gross_loss += abs(pnl)
                duration_hours = (i - entry_step) * 1.0  # 1 step = 1 hour
                trade_durations.append(duration_hours)
                trades.append(BacktestTrade(
                    timestamp=i, symbol=config.symbol, side="SELL",
                    quantity=position, price=exec_price,
                    commission=comm, slippage=slip, pnl=pnl,
                ))
                position = 0.0

        # Mark to market any open position
        if position > 0.0 and prices:
            final_price = prices[-1]
            capital += position * final_price

        # Equity curve and metrics
        equity_curve = self.compute_equity_curve(trades, config.initial_capital)
        # Ensure equity curve has at least n points (fill forward from trades)
        equity_curve_full = self._build_full_equity_curve(
            trades, config.initial_capital, n
        )

        total_ret = (capital - config.initial_capital) / config.initial_capital
        # Annualise: assume 8760 hours per year, each step = 1 hour
        n_years = max(n / 8760.0, 1 / 8760.0)
        ann_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        returns = [
            (equity_curve_full[i] - equity_curve_full[i - 1]) / equity_curve_full[i - 1]
            if equity_curve_full[i - 1] != 0 else 0.0
            for i in range(1, len(equity_curve_full))
        ]
        sr = self.sharpe_ratio(returns)
        mdd = self.max_drawdown(equity_curve_full)
        num_sells = sum(1 for t in trades if t.side == "SELL")
        win_rate = (winning_trades / num_sells) if num_sells > 0 else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_duration = (sum(trade_durations) / len(trade_durations)) if trade_durations else 0.0

        return BacktestMetrics(
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe_ratio=sr,
            max_drawdown=mdd,
            win_rate=win_rate,
            profit_factor=pf,
            num_trades=len(trades),
            avg_trade_duration_hours=avg_duration,
        )

    def compute_equity_curve(
        self, trades: List[BacktestTrade], initial_capital: float
    ) -> List[float]:
        """Build equity curve from trade list (cumulative PnL)."""
        curve = [initial_capital]
        cumulative = initial_capital
        for trade in trades:
            if trade.side == "SELL":
                cumulative += trade.pnl
            curve.append(cumulative)
        return curve

    def _build_full_equity_curve(
        self, trades: List[BacktestTrade], initial_capital: float, n: int
    ) -> List[float]:
        """Build an equity curve with one point per step."""
        curve = [initial_capital] * n
        cumulative = initial_capital
        trade_iter = iter(trades)
        trade = next(trade_iter, None)
        for i in range(n):
            while trade is not None and trade.timestamp == i:
                if trade.side == "SELL":
                    cumulative += trade.pnl
                trade = next(trade_iter, None)
            curve[i] = cumulative
        return curve

    def compute_drawdown(self, equity_curve: List[float]) -> List[float]:
        """Compute the drawdown at each point as a fraction."""
        drawdowns = []
        peak = equity_curve[0] if equity_curve else 1.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            drawdowns.append(dd)
        return drawdowns

    def max_drawdown(self, equity_curve: List[float]) -> float:
        """Maximum drawdown over the equity curve."""
        if not equity_curve:
            return 0.0
        dd = self.compute_drawdown(equity_curve)
        return max(dd) if dd else 0.0

    def sharpe_ratio(self, returns: List[float], risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio (assumes hourly returns)."""
        if len(returns) < 2:
            return 0.0
        excess = [r - risk_free for r in returns]
        mean_ex = sum(excess) / len(excess)
        variance = sum((r - mean_ex) ** 2 for r in excess) / (len(excess) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0.0:
            return 0.0
        # Annualise: sqrt(8760) for hourly data
        return (mean_ex / std) * math.sqrt(8760)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestEngine(unittest.TestCase):
    def _make_config(self) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=10_000.0,
            commission_bps=10.0,
            slippage_bps=5.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            symbol="BTC/USDT",
        )

    def test_run_no_signals(self):
        engine = BacktestEngine()
        prices = [100.0] * 50
        signals = [0] * 50
        metrics = engine.run(prices, signals, self._make_config())
        self.assertEqual(metrics.num_trades, 0)
        self.assertAlmostEqual(metrics.total_return, 0.0, places=5)

    def test_run_buy_and_sell(self):
        engine = BacktestEngine()
        prices = [100.0, 100.0, 110.0, 110.0]
        signals = [1, 0, -1, 0]
        metrics = engine.run(prices, signals, self._make_config())
        self.assertEqual(metrics.num_trades, 2)
        self.assertGreater(metrics.total_return, 0.0)

    def test_run_losing_trade(self):
        engine = BacktestEngine()
        prices = [100.0, 100.0, 80.0, 80.0]
        signals = [1, 0, -1, 0]
        metrics = engine.run(prices, signals, self._make_config())
        self.assertLess(metrics.total_return, 0.0)

    def test_compute_equity_curve_length(self):
        engine = BacktestEngine()
        trades = [
            BacktestTrade(0, "BTC", "BUY", 1.0, 100.0, 0.1, 0.05, 0.0),
            BacktestTrade(1, "BTC", "SELL", 1.0, 110.0, 0.1, 0.05, 9.75),
        ]
        curve = engine.compute_equity_curve(trades, 10000.0)
        self.assertEqual(len(curve), 3)

    def test_compute_equity_curve_values(self):
        engine = BacktestEngine()
        trades = [
            BacktestTrade(0, "BTC", "SELL", 1.0, 100.0, 0.0, 0.0, 50.0),
        ]
        curve = engine.compute_equity_curve(trades, 1000.0)
        self.assertAlmostEqual(curve[-1], 1050.0)

    def test_compute_drawdown_flat(self):
        engine = BacktestEngine()
        dd = engine.compute_drawdown([100.0, 100.0, 100.0])
        for d in dd:
            self.assertAlmostEqual(d, 0.0)

    def test_compute_drawdown_declining(self):
        engine = BacktestEngine()
        dd = engine.compute_drawdown([100.0, 90.0, 80.0])
        self.assertAlmostEqual(dd[0], 0.0)
        self.assertGreater(dd[2], dd[1])

    def test_max_drawdown_zero_for_monotone(self):
        engine = BacktestEngine()
        mdd = engine.max_drawdown([100.0, 110.0, 120.0])
        self.assertAlmostEqual(mdd, 0.0)

    def test_max_drawdown_detects_drop(self):
        engine = BacktestEngine()
        mdd = engine.max_drawdown([100.0, 90.0, 80.0, 90.0])
        self.assertAlmostEqual(mdd, 0.2, places=5)

    def test_sharpe_ratio_positive(self):
        engine = BacktestEngine()
        # Varying positive returns so std > 0
        returns = [0.001 * (1 + 0.1 * (i % 5)) for i in range(100)]
        sr = engine.sharpe_ratio(returns)
        self.assertGreater(sr, 0.0)

    def test_sharpe_ratio_zero_for_flat(self):
        engine = BacktestEngine()
        sr = engine.sharpe_ratio([0.0] * 50)
        self.assertAlmostEqual(sr, 0.0)

    def test_sharpe_ratio_empty(self):
        engine = BacktestEngine()
        sr = engine.sharpe_ratio([])
        self.assertAlmostEqual(sr, 0.0)

    def test_win_rate_all_winners(self):
        engine = BacktestEngine()
        prices = [100.0, 100.0, 110.0, 110.0, 120.0, 120.0, 130.0]
        signals = [1, 0, -1, 1, -1, 1, -1]
        metrics = engine.run(prices, signals, self._make_config())
        self.assertGreater(metrics.win_rate, 0.0)

    def test_metrics_fields_present(self):
        engine = BacktestEngine()
        metrics = engine.run([100.0] * 10, [0] * 10, self._make_config())
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)


if __name__ == "__main__":
    unittest.main()

"""Walk-forward backtesting with rolling train/test windows.

Walk-forward analysis is the gold standard for avoiding look-ahead bias in
algorithmic trading research.  Instead of training once on all history and
testing on a single held-out period, this module:

1. Splits a price series into overlapping (train, test) windows.
2. Trains a fresh agent on each train window.
3. Evaluates the agent on the immediately following test window.
4. Slides the window forward and repeats.

The result is a sequence of out-of-sample equity curves that, when stitched
together, form a realistic performance estimate free from look-ahead bias.

Usage::

    from walk_forward import WalkForwardEngine, WalkForwardConfig
    import pandas as pd

    df = pd.read_csv("data/BTC_prices.csv", parse_dates=["timestamp"], index_col="timestamp")

    config = WalkForwardConfig(
        train_periods=180,   # 180 bars per training window
        test_periods=30,     # evaluate on next 30 bars
        step=30,             # slide by 30 bars each iteration
        n_jobs=1,            # parallel folds (set >1 if RAM allows)
    )

    engine = WalkForwardEngine(config=config)
    results = engine.run(df, price_col="close")
    print(results.summary())

    results.plot_equity_curve("walk_forward_equity.png")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward run.

    Attributes:
        train_periods: Number of bars in each training window.
        test_periods:  Number of bars in each test (evaluation) window.
        step:          Bars to advance the window after each fold.
                       ``step < test_periods`` creates overlapping test windows.
                       ``step == test_periods`` creates non-overlapping windows.
        min_folds:     Minimum number of folds required; raises if fewer available.
        n_jobs:        Number of parallel workers (1 = serial).  Requires
                       ``multiprocessing`` to be safe (agent must be picklable).
        initial_cash:  Starting portfolio cash for each fold simulation.
        commission:    Round-trip commission rate (e.g. 0.001 = 0.1%).
        slippage:      Slippage per trade as a fraction of price.
        seed:          Global random seed for reproducibility.
    """

    train_periods: int = 120
    test_periods: int = 30
    step: int = 30
    min_folds: int = 3
    n_jobs: int = 1
    initial_cash: float = 10_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    seed: int = 42


@dataclass
class FoldResult:
    """Performance metrics for a single walk-forward fold.

    Attributes:
        fold_index:       Zero-based fold number.
        train_start:      First bar index of the training window.
        train_end:        Last bar index of the training window (exclusive).
        test_start:       First bar index of the test window.
        test_end:         Last bar index of the test window (exclusive).
        equity_curve:     Portfolio value at every test bar.
        total_return:     Fractional total return over the test window.
        sharpe:           Annualised Sharpe ratio (assumes daily bars).
        max_drawdown:     Maximum peak-to-trough drawdown (negative).
        num_trades:       Number of round-trip trades executed.
        win_rate:         Fraction of trades that were profitable.
        elapsed_secs:     Wall-clock time for this fold in seconds.
        extra:            Any additional metrics from the agent/strategy.
    """

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    equity_curve: np.ndarray
    total_return: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    elapsed_secs: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes:
        folds:        Individual fold results.
        config:       Configuration used for the run.
        price_index:  DatetimeIndex / RangeIndex of the original price series.
    """

    folds: list[FoldResult]
    config: WalkForwardConfig
    price_index: Any  # pd.Index

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def mean_sharpe(self) -> float:
        """Mean Sharpe ratio across folds."""
        if not self.folds:
            return float("nan")
        return float(np.mean([f.sharpe for f in self.folds]))

    def mean_return(self) -> float:
        """Mean total return across folds."""
        if not self.folds:
            return float("nan")
        return float(np.mean([f.total_return for f in self.folds]))

    def mean_max_drawdown(self) -> float:
        """Mean maximum drawdown across folds."""
        if not self.folds:
            return float("nan")
        return float(np.mean([f.max_drawdown for f in self.folds]))

    def mean_win_rate(self) -> float:
        """Mean win rate across folds."""
        if not self.folds:
            return float("nan")
        return float(np.mean([f.win_rate for f in self.folds]))

    def stitched_equity_curve(self) -> np.ndarray:
        """Concatenate test equity curves into a single out-of-sample curve."""
        if not self.folds:
            return np.array([])
        curves = [f.equity_curve for f in sorted(self.folds, key=lambda f: f.fold_index)]
        # Chain curves: rescale each to start where the previous ended
        result = [curves[0]]
        for curve in curves[1:]:
            scale = result[-1][-1] / curve[0] if curve[0] != 0.0 else 1.0
            result.append(curve * scale)
        return np.concatenate(result)

    def summary(self) -> str:
        """Human-readable summary table."""
        lines: list[str] = [
            "Walk-Forward Summary",
            "=" * 50,
            f"  Folds:              {len(self.folds)}",
            f"  Mean Sharpe:        {self.mean_sharpe():.3f}",
            f"  Mean Return:        {self.mean_return() * 100:.2f}%",
            f"  Mean Max Drawdown:  {self.mean_max_drawdown() * 100:.2f}%",
            f"  Mean Win Rate:      {self.mean_win_rate() * 100:.1f}%",
            "",
            "Per-Fold Results:",
        ]
        for f in sorted(self.folds, key=lambda x: x.fold_index):
            lines.append(
                f"  Fold {f.fold_index:3d} | "
                f"train [{f.train_start:5d},{f.train_end:5d}) | "
                f"test [{f.test_start:5d},{f.test_end:5d}) | "
                f"sharpe={f.sharpe:6.3f} | "
                f"ret={f.total_return * 100:+6.2f}% | "
                f"mdd={f.max_drawdown * 100:+6.2f}% | "
                f"trades={f.num_trades:4d}"
            )
        return "\n".join(lines)

    def plot_equity_curve(
        self,
        output_path: str | Path = "walk_forward_equity.png",
        title: str = "Walk-Forward Out-of-Sample Equity Curve",
    ) -> None:
        """Save a plot of the stitched equity curve to *output_path*."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot")
            return

        curve = self.stitched_equity_curve()
        if len(curve) == 0:
            logger.warning("No equity data to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Top: equity curve
        ax_eq = axes[0]
        ax_eq.plot(curve, linewidth=1.5, color="#2196F3", label="Stitched OOS equity")
        ax_eq.axhline(y=curve[0], color="gray", linestyle="--", linewidth=0.8)
        ax_eq.set_title(title, fontsize=14)
        ax_eq.set_ylabel("Portfolio Value ($)")
        ax_eq.grid(True, alpha=0.3)
        ax_eq.legend()

        # Shade fold boundaries
        colors = ["#E3F2FD", "#FFF8E1"]
        pos = 0
        for i, fold in enumerate(sorted(self.folds, key=lambda x: x.fold_index)):
            fold_len = len(fold.equity_curve)
            ax_eq.axvspan(pos, pos + fold_len, alpha=0.15, color=colors[i % 2])
            pos += fold_len

        # Bottom: drawdown
        ax_dd = axes[1]
        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak
        ax_dd.fill_between(range(len(drawdown)), drawdown, 0, color="#EF5350", alpha=0.7)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Test Bar")
        ax_dd.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Walk-forward equity curve saved to %s", output_path)


# ---------------------------------------------------------------------------
# Built-in simple strategy (momentum) — used as the default agent
# ---------------------------------------------------------------------------


class _MomentumStrategy:
    """Simple EMA-crossover momentum strategy used when no custom agent is provided."""

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        self.fast = fast
        self.slow = slow

    def fit(self, train_prices: np.ndarray) -> None:
        """No fitting required for a rule-based strategy."""
        pass

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Return +1 (long), -1 (short), or 0 (flat) for each bar."""
        def ema(x: np.ndarray, n: int) -> np.ndarray:
            alpha = 2.0 / (n + 1)
            result = np.empty_like(x, dtype=float)
            result[0] = x[0]
            for i in range(1, len(x)):
                result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
            return result

        if len(prices) < self.slow:
            return np.zeros(len(prices))
        fast_ema = ema(prices, self.fast)
        slow_ema = ema(prices, self.slow)
        positions = np.where(fast_ema > slow_ema, 1.0, -1.0)
        return positions


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class WalkForwardEngine:
    """Walk-forward backtesting engine.

    Parameters
    ----------
    config:
        Walk-forward configuration (window sizes, step, parallelism).
    strategy_factory:
        Callable that returns a fresh strategy/agent for each fold.
        The strategy must implement ``fit(train_prices)`` and
        ``predict(prices) -> np.ndarray``.
        Defaults to :class:`_MomentumStrategy` when ``None``.
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        strategy_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config or WalkForwardConfig()
        self.strategy_factory = strategy_factory or (lambda: _MomentumStrategy())

    def _generate_folds(
        self, n: int
    ) -> list[tuple[int, int, int, int]]:
        """Yield (train_start, train_end, test_start, test_end) index tuples."""
        folds: list[tuple[int, int, int, int]] = []
        train_periods = self.config.train_periods
        test_periods = self.config.test_periods
        step = self.config.step

        start = 0
        while True:
            train_end = start + train_periods
            test_end = train_end + test_periods
            if test_end > n:
                break
            folds.append((start, train_end, train_end, test_end))
            start += step

        return folds

    def _simulate_fold(
        self,
        fold_idx: int,
        train_prices: np.ndarray,
        test_prices: np.ndarray,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
    ) -> FoldResult:
        """Run a single fold: train → simulate on test → return metrics."""
        t0 = time.perf_counter()

        strategy = self.strategy_factory()

        # Fix random seeds for this fold for reproducibility
        rng_seed = self.config.seed + fold_idx
        np.random.seed(rng_seed % (2**32))
        try:
            import torch
            torch.manual_seed(rng_seed % (2**32))
        except ImportError:
            pass

        # Train
        strategy.fit(train_prices)

        # Generate signals on test period
        signals = strategy.predict(test_prices)
        signals = np.clip(signals, -1.0, 1.0)

        # Simulate PnL
        equity_curve, trade_returns = self._simulate_pnl(test_prices, signals)

        # Compute metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = self._annualised_sharpe(daily_returns)
        max_dd = self._max_drawdown(equity_curve)
        num_trades = max(1, len(trade_returns))
        win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0

        elapsed = time.perf_counter() - t0

        return FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            equity_curve=equity_curve,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            num_trades=num_trades,
            win_rate=win_rate,
            elapsed_secs=elapsed,
        )

    def _simulate_pnl(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """Simulate portfolio returns from signals.

        Returns
        -------
        equity_curve : np.ndarray
            Portfolio value at each bar.
        trade_returns : list[float]
            Fractional return for each completed round-trip trade.
        """
        cash = self.config.initial_cash
        equity_curve = np.empty(len(prices))
        equity_curve[0] = cash
        position = 0.0  # units held (+ long, - short)
        entry_price = 0.0
        trade_returns: list[float] = []
        commission = self.config.commission
        slippage = self.config.slippage

        for i in range(1, len(prices)):
            price = prices[i]
            signal = signals[i - 1]  # signal at previous bar drives current action

            desired_position_sign = np.sign(signal) if abs(signal) > 0.05 else 0.0

            if desired_position_sign != np.sign(position):
                # Close existing position
                if position != 0.0:
                    exit_price = price * (1 - slippage * np.sign(position))
                    pnl = position * (exit_price - entry_price)
                    fee = abs(position) * exit_price * commission
                    cash += pnl - fee
                    ret = (exit_price - entry_price) / entry_price if entry_price != 0 else 0.0
                    trade_returns.append(float(ret * np.sign(position)))
                    position = 0.0

                # Open new position
                if desired_position_sign != 0.0 and cash > 0:
                    entry_price = price * (1 + slippage * desired_position_sign)
                    units = (cash * 0.95) / entry_price  # use 95% of cash
                    position = units * desired_position_sign
                    fee = abs(position) * entry_price * commission
                    cash -= fee

            # Mark-to-market
            if position != 0.0:
                mtm = position * (price - entry_price)
                equity_curve[i] = cash + mtm + abs(position) * entry_price
            else:
                equity_curve[i] = cash

        return equity_curve, trade_returns

    @staticmethod
    def _annualised_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        if sigma == 0.0:
            return 0.0
        return float(mu / sigma * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.where(peak == 0, 1.0, peak)
        return float(np.min(drawdown))

    def run(
        self,
        data: pd.DataFrame | np.ndarray,
        price_col: str = "close",
    ) -> WalkForwardResult:
        """Execute the full walk-forward run.

        Parameters
        ----------
        data:
            DataFrame with a price column, or a 1-D numpy array of prices.
        price_col:
            Column name to use as the price series when *data* is a DataFrame.

        Returns
        -------
        WalkForwardResult
            Aggregated results across all folds.
        """
        if isinstance(data, pd.DataFrame):
            price_index = data.index
            prices = data[price_col].to_numpy(dtype=float)
        else:
            prices = np.asarray(data, dtype=float)
            price_index = np.arange(len(prices))

        n = len(prices)
        folds_spec = self._generate_folds(n)

        if len(folds_spec) < self.config.min_folds:
            raise ValueError(
                f"Only {len(folds_spec)} fold(s) possible with {n} bars, "
                f"but min_folds={self.config.min_folds}. "
                f"Reduce train_periods ({self.config.train_periods}), "
                f"test_periods ({self.config.test_periods}), or min_folds."
            )

        logger.info(
            "Walk-forward: %d bars, %d folds "
            "(train=%d, test=%d, step=%d)",
            n, len(folds_spec),
            self.config.train_periods, self.config.test_periods, self.config.step,
        )

        fold_results: list[FoldResult] = []

        if self.config.n_jobs == 1:
            # Serial execution
            for idx, (ts, te, vs, ve) in enumerate(folds_spec):
                result = self._simulate_fold(
                    idx,
                    prices[ts:te],
                    prices[vs:ve],
                    ts, te, vs, ve,
                )
                fold_results.append(result)
                logger.info(
                    "Fold %d/%d done — sharpe=%.3f ret=%.2f%% dd=%.2f%%",
                    idx + 1, len(folds_spec),
                    result.sharpe,
                    result.total_return * 100,
                    result.max_drawdown * 100,
                )
        else:
            # Parallel execution via ProcessPoolExecutor
            futures: dict = {}
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as pool:
                for idx, (ts, te, vs, ve) in enumerate(folds_spec):
                    future = pool.submit(
                        self._simulate_fold,
                        idx,
                        prices[ts:te],
                        prices[vs:ve],
                        ts, te, vs, ve,
                    )
                    futures[future] = idx
                for future in as_completed(futures):
                    result = future.result()
                    fold_results.append(result)
                    logger.info(
                        "Fold %d/%d done — sharpe=%.3f ret=%.2f%%",
                        futures[future] + 1, len(folds_spec),
                        result.sharpe,
                        result.total_return * 100,
                    )
            fold_results.sort(key=lambda r: r.fold_index)

        return WalkForwardResult(
            folds=fold_results,
            config=self.config,
            price_index=price_index,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting engine for LARSA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="CSV file with price data")
    parser.add_argument("--price-col", default="close", help="Price column name")
    parser.add_argument("--train", type=int, default=120, help="Training window (bars)")
    parser.add_argument("--test", type=int, default=30, help="Test window (bars)")
    parser.add_argument("--step", type=int, default=30, help="Window advance step (bars)")
    parser.add_argument("--min-folds", type=int, default=3, help="Minimum folds required")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel fold workers")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Starting cash per fold")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--plot", default="walk_forward_equity.png", help="Output equity curve image"
    )
    return parser


def main() -> None:
    """CLI entry point."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    if args.price_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        print(
            f"Error: column '{args.price_col}' not in CSV. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    config = WalkForwardConfig(
        train_periods=args.train,
        test_periods=args.test,
        step=args.step,
        min_folds=args.min_folds,
        n_jobs=args.jobs,
        initial_cash=args.cash,
        commission=args.commission,
        seed=args.seed,
    )

    engine = WalkForwardEngine(config=config)
    results = engine.run(df, price_col=args.price_col)

    print(results.summary())
    results.plot_equity_curve(args.plot)
    print(f"\nEquity curve plot saved to: {args.plot}")


if __name__ == "__main__":
    main()

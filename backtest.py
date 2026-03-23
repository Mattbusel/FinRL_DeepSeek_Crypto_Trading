"""Backtesting engine for the LARSA trading system.

Loads historical OHLCV data from a CSV file, runs a momentum agent through the
price series, computes performance metrics (Sharpe ratio, max drawdown, win
rate, total return), and saves an equity curve chart.

Usage::

    python backtest.py --csv ./data/BTC_1sec_with_sentiment_risk_train.csv
    python backtest.py --csv ./data/prices.csv --cash 50000 --plot equity.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logger import get_logger
from metrics import sharpe_ratio, max_drawdown, cumulative_returns

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Container for backtesting performance metrics.

    Attributes:
        total_return: Total percentage return over the period.
        sharpe: Sharpe ratio (non-annualised).
        max_dd: Maximum drawdown (negative float, e.g. -0.25 = -25%).
        win_rate: Fraction of trades that were profitable.
        num_trades: Total number of executed trades.
        equity_curve: Portfolio value at each step.
        trade_returns: Per-trade return values.
    """

    total_return: float
    sharpe: float
    max_dd: float
    win_rate: float
    num_trades: int
    equity_curve: list[float] = field(default_factory=list)
    trade_returns: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_PRICE_COLS = ["close", "Close", "price", "Price", "btc_close", "BTC_Close"]


def _detect_price_column(df: pd.DataFrame) -> str:
    """Return the name of the price column in *df*.

    Tries a list of common column names in priority order.

    Args:
        df: The loaded DataFrame.

    Returns:
        Column name string.

    Raises:
        ValueError: If no recognised price column is found.
    """
    for col in _PRICE_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot find a price column. Expected one of {_PRICE_COLS}; "
        f"got columns: {list(df.columns)[:10]}"
    )


def load_ohlcv(csv_path: str, price_col: str | None = None) -> pd.Series:
    """Load a price series from a CSV file.

    Args:
        csv_path: Path to the OHLCV CSV.
        price_col: Explicit column name; auto-detected when ``None``.

    Returns:
        A :class:`pandas.Series` of prices with a numeric index.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If no price column can be found.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info("backtest.load_csv", path=csv_path)
    df = pd.read_csv(csv_path)
    col = price_col or _detect_price_column(df)
    log.info("backtest.price_col", column=col, rows=len(df))

    prices = df[col].dropna().reset_index(drop=True).astype(float)
    return prices


# ---------------------------------------------------------------------------
# Momentum agent (same as paper_trade.py, kept self-contained)
# ---------------------------------------------------------------------------


class _MomentumAgent:
    """EMA crossover momentum agent for backtesting."""

    def __init__(self, short: int = 5, long: int = 20) -> None:
        self._short = short
        self._long = long
        self._prices: list[float] = []
        self._prev: int | None = None

    def _ema(self, arr: list[float], span: int) -> float:
        if not arr:
            return 0.0
        alpha = 2.0 / (span + 1)
        ema = arr[0]
        for v in arr[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def act(self, price: float) -> int:
        """Return action: 0 = hold, 1 = buy, 2 = sell."""
        self._prices.append(price)
        if len(self._prices) < self._long:
            return 0
        recent = self._prices[-self._long :]
        sig = 1 if self._ema(recent[-self._short :], self._short) > self._ema(recent, self._long) else -1
        action = 0
        if self._prev == -1 and sig == 1:
            action = 1
        elif self._prev == 1 and sig == -1:
            action = 2
        self._prev = sig
        return action


# ---------------------------------------------------------------------------
# Backtesting loop
# ---------------------------------------------------------------------------


def run_backtest(
    prices: pd.Series,
    starting_cash: float = 100_000.0,
    slippage: float = 0.0005,
    trade_fraction: float = 0.10,
    short_window: int = 5,
    long_window: int = 20,
) -> BacktestResult:
    """Run the backtesting loop over a price series.

    Args:
        prices: Chronologically ordered price series (USD).
        starting_cash: Starting portfolio cash.
        slippage: Per-trade execution slippage fraction.
        trade_fraction: Fraction of cash/holdings traded per signal.
        short_window: Fast EMA period.
        long_window: Slow EMA period.

    Returns:
        A populated :class:`BacktestResult`.
    """
    cash = starting_cash
    btc_held = 0.0
    agent = _MomentumAgent(short=short_window, long=long_window)

    equity_curve: list[float] = []
    trade_pnl_list: list[float] = []  # P&L per closed trade
    open_entry: float | None = None  # entry price of current BTC position

    for price in prices:
        action = agent.act(float(price))

        if action == 1:  # BUY
            spend = cash * trade_fraction
            if spend >= 1.0:
                exec_price = price * (1 + slippage)
                btc_bought = spend / exec_price
                cash -= spend
                btc_held += btc_bought
                if open_entry is None:
                    open_entry = exec_price

        elif action == 2:  # SELL
            sell_qty = btc_held * trade_fraction
            if sell_qty >= 1e-8:
                exec_price = price * (1 - slippage)
                usd_received = sell_qty * exec_price
                if open_entry is not None:
                    trade_pnl_list.append((exec_price - open_entry) / open_entry)
                    open_entry = None
                btc_held -= sell_qty
                cash += usd_received

        equity_curve.append(cash + btc_held * price)

    # Compute summary metrics
    final_value = equity_curve[-1] if equity_curve else starting_cash
    total_return = (final_value - starting_cash) / starting_cash * 100.0

    # Per-step returns for Sharpe / drawdown
    pct_returns: np.ndarray = np.diff(np.array(equity_curve)) / np.array(equity_curve[:-1])
    if pct_returns.size > 1:
        sr = sharpe_ratio(pct_returns)
        mdd = max_drawdown(pct_returns)
    else:
        sr = float("nan")
        mdd = 0.0

    win_rate = (
        float(np.sum(np.array(trade_pnl_list) > 0)) / len(trade_pnl_list)
        if trade_pnl_list
        else float("nan")
    )

    return BacktestResult(
        total_return=total_return,
        sharpe=sr,
        max_dd=mdd,
        win_rate=win_rate,
        num_trades=len(trade_pnl_list),
        equity_curve=equity_curve,
        trade_returns=trade_pnl_list,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: BacktestResult) -> None:
    """Print a human-readable backtest summary to stdout."""
    print("\n" + "=" * 60)
    print("  LARSA Backtesting Report")
    print("=" * 60)
    print(f"  Total Return  : {result.total_return:+.2f}%")
    print(f"  Sharpe Ratio  : {result.sharpe:.4f}")
    print(f"  Max Drawdown  : {result.max_dd * 100:.2f}%")
    print(f"  Win Rate      : {result.win_rate * 100:.1f}%" if not _isnan(result.win_rate) else "  Win Rate      : N/A")
    print(f"  Num Trades    : {result.num_trades}")
    print("=" * 60 + "\n")


def _isnan(v: float) -> bool:
    import math
    return math.isnan(v)


def save_equity_chart(result: BacktestResult, output_path: str = "equity_curve.png") -> None:
    """Generate and save an equity curve chart.

    Args:
        result: Completed :class:`BacktestResult`.
        output_path: File path for the PNG chart.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- Equity curve ---
    ax1 = axes[0]
    equity = np.array(result.equity_curve)
    ax1.plot(equity, color="#2196F3", linewidth=1.2, label="Portfolio Value")
    ax1.set_title("LARSA Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Annotate final value
    final = equity[-1] if len(equity) > 0 else 0
    ax1.annotate(
        f"${final:,.0f}",
        xy=(len(equity) - 1, final),
        xytext=(-80, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
    )

    # --- Drawdown ---
    ax2 = axes[1]
    if len(equity) > 1:
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#EF5350", alpha=0.6, label="Drawdown %")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Steps")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # Subtitle with metrics
    fig.suptitle(
        f"Total Return: {result.total_return:+.2f}%  |  "
        f"Sharpe: {result.sharpe:.3f}  |  "
        f"Max DD: {result.max_dd * 100:.2f}%  |  "
        f"Trades: {result.num_trades}",
        fontsize=10,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("backtest.chart_saved", path=output_path)
    print(f"  Equity curve saved to: {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the backtesting engine."""
    parser = argparse.ArgumentParser(
        description="LARSA backtesting engine — runs agent over historical OHLCV CSV data."
    )
    parser.add_argument(
        "--csv",
        default="./data/BTC_1sec_with_sentiment_risk_train.csv",
        help="Path to historical OHLCV CSV file.",
    )
    parser.add_argument("--price-col", default=None, help="CSV column name for price (auto-detected if omitted).")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash in USD.")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Per-trade slippage fraction.")
    parser.add_argument("--trade-fraction", type=float, default=0.10, help="Fraction of holdings per trade.")
    parser.add_argument("--short-window", type=int, default=5, help="Fast EMA window (ticks).")
    parser.add_argument("--long-window", type=int, default=20, help="Slow EMA window (ticks).")
    parser.add_argument("--plot", default="equity_curve.png", help="Output path for equity curve chart.")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation.")
    args = parser.parse_args()

    prices = load_ohlcv(args.csv, price_col=args.price_col)
    log.info("backtest.start", rows=len(prices), starting_cash=args.cash)

    result = run_backtest(
        prices=prices,
        starting_cash=args.cash,
        slippage=args.slippage,
        trade_fraction=args.trade_fraction,
        short_window=args.short_window,
        long_window=args.long_window,
    )

    print_report(result)

    if not args.no_plot:
        save_equity_chart(result, output_path=args.plot)


if __name__ == "__main__":
    main()

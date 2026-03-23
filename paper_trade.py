"""Paper trading mode for the LARSA trading system.

Connects to the Binance public REST API (no authentication required) to fetch
live BTC/USDT prices, runs the DQN agent against those prices, tracks a
virtual portfolio, logs every trade decision to ``paper_trades.jsonl``, and
prints real-time performance to the terminal.

Usage::

    python paper_trade.py                        # default 100 000 USD starting cash
    python paper_trade.py --cash 50000           # custom starting balance
    python paper_trade.py --interval 5 --steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"
LOG_FILE = "paper_trades.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VirtualPortfolio:
    """Tracks the virtual portfolio state during paper trading.

    Attributes:
        cash: Available USD cash.
        btc_held: BTC units currently held.
        starting_cash: Initial cash balance (for P&L calculation).
        trades: List of executed virtual trade records.
    """

    cash: float
    btc_held: float = 0.0
    starting_cash: float = field(init=False)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.starting_cash = self.cash

    def portfolio_value(self, price: float) -> float:
        """Total portfolio value in USD at the given BTC price."""
        return self.cash + self.btc_held * price

    def pnl(self, price: float) -> float:
        """Absolute P&L relative to starting cash."""
        return self.portfolio_value(price) - self.starting_cash

    def pnl_pct(self, price: float) -> float:
        """Percentage P&L relative to starting cash."""
        if self.starting_cash == 0:
            return 0.0
        return self.pnl(price) / self.starting_cash * 100.0


# ---------------------------------------------------------------------------
# Binance data fetch
# ---------------------------------------------------------------------------


def fetch_live_price(symbol: str = SYMBOL) -> float | None:
    """Fetch the latest BTC/USDT price from the Binance public REST API.

    Args:
        symbol: Trading pair symbol.

    Returns:
        Current price as a float, or ``None`` if the request fails.
    """
    try:
        resp = requests.get(
            BINANCE_TICKER_URL,
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"])
    except Exception as exc:
        log.warning("binance.fetch_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Simple rule-based agent (no model checkpoint required)
# ---------------------------------------------------------------------------


class SimpleMomentumAgent:
    """Minimal momentum agent usable without a trained model checkpoint.

    Tracks a short exponential moving average and a longer one.  Buys when
    the short EMA crosses above the long EMA and sells when it crosses below.
    Action space mirrors the DQN conventions: 0 = hold, 1 = buy, 2 = sell.

    Args:
        short_window: EMA span for the fast signal (in ticks).
        long_window: EMA span for the slow signal (in ticks).
    """

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self._prices: list[float] = []
        self._prev_signal: int | None = None  # 1 = short > long, -1 = short < long

    def _ema(self, prices: list[float], span: int) -> float:
        """Compute a simple exponential moving average."""
        if not prices:
            return 0.0
        alpha = 2.0 / (span + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def act(self, price: float) -> int:
        """Return an action given the latest price tick.

        Args:
            price: Current BTC price.

        Returns:
            Action integer: 0 = hold, 1 = buy, 2 = sell.
        """
        self._prices.append(price)
        if len(self._prices) < self.long_window:
            return 0  # not enough data yet, hold

        recent = self._prices[-self.long_window :]
        short_ema = self._ema(recent[-self.short_window :], self.short_window)
        long_ema = self._ema(recent, self.long_window)

        current_signal = 1 if short_ema > long_ema else -1

        action = 0  # default hold
        if self._prev_signal == -1 and current_signal == 1:
            action = 1  # crossover up → buy
        elif self._prev_signal == 1 and current_signal == -1:
            action = 2  # crossover down → sell

        self._prev_signal = current_signal
        return action


# ---------------------------------------------------------------------------
# Trade logger
# ---------------------------------------------------------------------------


def _log_trade(record: dict[str, Any], log_file: str = LOG_FILE) -> None:
    """Append a trade record to the JSONL log file.

    Args:
        record: Dictionary of trade fields.
        log_file: Path to the JSONL output file.
    """
    try:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.error("trade_log.write_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

ACTION_NAMES = {0: "HOLD", 1: "BUY ", 2: "SELL"}


def _execute_action(
    action: int,
    price: float,
    portfolio: VirtualPortfolio,
    slippage: float = 0.0005,
    trade_fraction: float = 0.10,
) -> dict[str, Any] | None:
    """Apply a trade action to the virtual portfolio.

    Args:
        action: 0 = hold, 1 = buy, 2 = sell.
        price: Current BTC price.
        portfolio: Mutable virtual portfolio instance.
        slippage: Per-trade slippage fraction applied to execution price.
        trade_fraction: Fraction of cash/holdings used per trade.

    Returns:
        A trade record dict if a trade was executed, else ``None``.
    """
    now_ts = datetime.now(timezone.utc).isoformat()

    if action == 1:  # BUY
        spend = portfolio.cash * trade_fraction
        if spend < 1.0:
            return None
        exec_price = price * (1 + slippage)
        btc_bought = spend / exec_price
        portfolio.cash -= spend
        portfolio.btc_held += btc_bought
        record = {
            "timestamp": now_ts,
            "action": "BUY",
            "price": exec_price,
            "btc_qty": round(btc_bought, 8),
            "usd_spent": round(spend, 4),
            "cash_remaining": round(portfolio.cash, 4),
            "btc_held": round(portfolio.btc_held, 8),
        }
        _log_trade(record)
        return record

    elif action == 2:  # SELL
        sell_qty = portfolio.btc_held * trade_fraction
        if sell_qty < 1e-8:
            return None
        exec_price = price * (1 - slippage)
        usd_received = sell_qty * exec_price
        portfolio.btc_held -= sell_qty
        portfolio.cash += usd_received
        record = {
            "timestamp": now_ts,
            "action": "SELL",
            "price": exec_price,
            "btc_qty": round(sell_qty, 8),
            "usd_received": round(usd_received, 4),
            "cash_remaining": round(portfolio.cash, 4),
            "btc_held": round(portfolio.btc_held, 8),
        }
        _log_trade(record)
        return record

    return None  # HOLD


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_status(
    step: int,
    price: float,
    action: int,
    portfolio: VirtualPortfolio,
    trade: dict[str, Any] | None,
) -> None:
    """Print a one-line real-time status update to the terminal."""
    pnl = portfolio.pnl(price)
    pnl_pct = portfolio.pnl_pct(price)
    pv = portfolio.portfolio_value(price)
    sign = "+" if pnl >= 0 else ""
    trade_str = f" | {trade['action']}" if trade else ""
    print(
        f"[Step {step:5d}]  BTC: ${price:>10,.2f}"
        f"  Action: {ACTION_NAMES.get(action, '?')}"
        f"  PortfolioValue: ${pv:>12,.2f}"
        f"  P&L: {sign}{pnl_pct:.2f}%  ({sign}${pnl:,.2f})"
        f"{trade_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_paper_trading(
    starting_cash: float = 100_000.0,
    interval_seconds: float = 10.0,
    max_steps: int | None = None,
    log_file: str = LOG_FILE,
    short_window: int = 5,
    long_window: int = 20,
    slippage: float = 0.0005,
    trade_fraction: float = 0.10,
) -> None:
    """Run the paper trading loop.

    Fetches live Binance prices, runs the momentum agent, and tracks a virtual
    portfolio.  Trades are appended to *log_file* in JSONL format.

    Args:
        starting_cash: Initial USD balance for the virtual portfolio.
        interval_seconds: Seconds between each price poll.
        max_steps: Stop after this many steps (``None`` = run forever).
        log_file: Path for the JSONL trade log.
        short_window: Fast EMA window (ticks).
        long_window: Slow EMA window (ticks).
        slippage: Per-trade slippage fraction.
        trade_fraction: Fraction of holdings used per trade.
    """
    portfolio = VirtualPortfolio(cash=starting_cash)
    agent = SimpleMomentumAgent(short_window=short_window, long_window=long_window)

    print("=" * 80)
    print(f"  LARSA Paper Trader  |  BTC/USDT (Binance)  |  Starting cash: ${starting_cash:,.2f}")
    print(f"  Log file: {os.path.abspath(log_file)}")
    print(f"  EMA crossover: short={short_window} / long={long_window}")
    print("=" * 80)

    step = 0
    try:
        while True:
            if max_steps is not None and step >= max_steps:
                break

            price = fetch_live_price()
            if price is None:
                log.warning("paper_trade.skip_step", reason="price_fetch_failed")
                time.sleep(interval_seconds)
                continue

            action = agent.act(price)
            trade = _execute_action(
                action,
                price,
                portfolio,
                slippage=slippage,
                trade_fraction=trade_fraction,
            )
            _print_status(step + 1, price, action, portfolio, trade)

            step += 1
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Final summary:")
        price = fetch_live_price() or 0.0
        print(f"  Steps run  : {step}")
        print(f"  Final price: ${price:,.2f}")
        print(f"  Portfolio  : ${portfolio.portfolio_value(price):,.2f}")
        print(f"  Total P&L  : ${portfolio.pnl(price):,.2f}  ({portfolio.pnl_pct(price):.2f}%)")
        print(f"  Trades log : {os.path.abspath(log_file)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LARSA paper trading mode — live Binance prices, virtual portfolio."
    )
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting USD cash balance.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between price polls.")
    parser.add_argument("--steps", type=int, default=None, help="Max steps (None = run forever).")
    parser.add_argument("--log-file", default=LOG_FILE, help="JSONL trade log output path.")
    parser.add_argument("--short-window", type=int, default=5, help="Fast EMA window.")
    parser.add_argument("--long-window", type=int, default=20, help="Slow EMA window.")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Per-trade slippage.")
    parser.add_argument("--trade-fraction", type=float, default=0.10, help="Fraction traded per signal.")
    args = parser.parse_args()

    run_paper_trading(
        starting_cash=args.cash,
        interval_seconds=args.interval,
        max_steps=args.steps,
        log_file=args.log_file,
        short_window=args.short_window,
        long_window=args.long_window,
        slippage=args.slippage,
        trade_fraction=args.trade_fraction,
    )


if __name__ == "__main__":
    main()

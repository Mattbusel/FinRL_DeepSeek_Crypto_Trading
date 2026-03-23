"""Live paper trading mode for the LARSA trading system.

Simulates live trading without real money.  Key components:

* :class:`TradeLog` -- immutable record of a single simulated trade.
* :class:`PaperPortfolio` -- tracks simulated positions, P&L, and metrics in
  real time with SQLite persistence.
* :class:`PaperTrader` -- connects to a Binance WebSocket (or a mock price
  feed), receives live prices, and executes paper trades driven by the LARSA
  ensemble agents.
* :func:`compare_vs_backtest` -- runs paper trading alongside a backtest replay
  to produce a side-by-side performance comparison.
* :func:`print_dashboard` -- terminal dashboard showing current positions and P&L.

Environment variables::

    BINANCE_WS_URL    WebSocket URL (default wss://stream.binance.com:9443)
    PAPER_DB_PATH     SQLite path for trade log (default ./paper_trades.db)

Example::

    from paper_trading import PaperTrader, PaperPortfolio
    portfolio = PaperPortfolio(starting_cash=100_000.0)
    trader = PaperTrader(portfolio=portfolio, assets=["BTCUSDT"])
    trader.run(max_ticks=500)
    print_dashboard(portfolio)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WS_URL = "wss://stream.binance.com:9443/ws"
_DEFAULT_DB_PATH = "./paper_trades.db"


# ---------------------------------------------------------------------------
# TradeLog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeLog:
    """Immutable record of a single simulated trade.

    Attributes:
        timestamp: Unix timestamp of trade execution (UTC).
        asset: Asset symbol traded (e.g. ``"BTCUSDT"``).
        action: ``"buy"``, ``"sell"``, or ``"hold"``.
        price: Execution price.
        quantity: Units traded (positive).
        pnl: Realised P&L on this trade (0.0 for open trades).
        trade_id: Unique trade identifier.
    """

    timestamp: float
    asset: str
    action: str
    price: float
    quantity: float
    pnl: float = 0.0
    trade_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "asset": self.asset,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "pnl": self.pnl,
        }


# ---------------------------------------------------------------------------
# PaperPortfolio
# ---------------------------------------------------------------------------


class PaperPortfolio:
    """Tracks simulated positions, P&L, and performance metrics in real time.

    Persists all trades to SQLite so state survives restarts.

    Args:
        starting_cash: Initial simulated cash balance (USD).
        slippage: Per-trade slippage fraction.
        db_path: Path to the SQLite file used for trade log persistence.

    Attributes:
        cash: Current cash balance.
        positions: Units held per asset symbol.
        trade_log: In-memory list of :class:`TradeLog` entries.
    """

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        slippage: float = 5e-4,
        db_path: str | None = None,
    ) -> None:
        self.starting_cash = starting_cash
        self.slippage = slippage
        self.db_path = db_path or os.environ.get("PAPER_DB_PATH", _DEFAULT_DB_PATH)

        self.cash: float = starting_cash
        self.positions: dict[str, float] = {}
        self.avg_entry_prices: dict[str, float] = {}
        self.trade_log: list[TradeLog] = []

        self._peak_value: float = starting_cash
        self._trade_counter: int = 0

        self._init_db()
        log.info(
            "paper_portfolio.init",
            starting_cash=starting_cash,
            db_path=self.db_path,
        )

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the trade log table if it does not already exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id    TEXT PRIMARY KEY,
                    timestamp   REAL NOT NULL,
                    asset       TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    price       REAL NOT NULL,
                    quantity    REAL NOT NULL,
                    pnl         REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            conn.commit()
            conn.close()
            log.debug("paper_portfolio.db.init", path=self.db_path)
        except Exception as exc:
            log.warning("paper_portfolio.db.init_fail", error=str(exc))

    def _persist_trade(self, trade: TradeLog) -> None:
        """Write a single trade record to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO paper_trades VALUES (?,?,?,?,?,?,?)",
                (
                    trade.trade_id,
                    trade.timestamp,
                    trade.asset,
                    trade.action,
                    trade.price,
                    trade.quantity,
                    trade.pnl,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("paper_portfolio.db.persist_fail", error=str(exc))

    def load_from_db(self) -> int:
        """Reload historical trade records from SQLite into ``trade_log``.

        Returns:
            Number of trades loaded.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT trade_id, timestamp, asset, action, price, quantity, pnl "
                "FROM paper_trades ORDER BY timestamp"
            ).fetchall()
            conn.close()
            for row in rows:
                self.trade_log.append(
                    TradeLog(
                        trade_id=row[0],
                        timestamp=row[1],
                        asset=row[2],
                        action=row[3],
                        price=row[4],
                        quantity=row[5],
                        pnl=row[6],
                    )
                )
            log.info("paper_portfolio.db.loaded", count=len(rows))
            return len(rows)
        except Exception as exc:
            log.warning("paper_portfolio.db.load_fail", error=str(exc))
            return 0

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_buy(self, asset: str, price: float, quantity: float) -> TradeLog:
        """Execute a simulated buy order.

        Args:
            asset: Asset symbol.
            price: Current market price.
            quantity: Units to purchase.

        Returns:
            The :class:`TradeLog` entry for this trade.
        """
        cost = price * quantity * (1 + self.slippage)
        if cost > self.cash:
            quantity = self.cash / (price * (1 + self.slippage))
            cost = self.cash

        old_qty = self.positions.get(asset, 0.0)
        old_avg = self.avg_entry_prices.get(asset, price)
        new_qty = old_qty + quantity
        # Weighted average entry price
        self.avg_entry_prices[asset] = (old_qty * old_avg + quantity * price) / max(new_qty, 1e-10)
        self.positions[asset] = new_qty
        self.cash -= cost

        self._trade_counter += 1
        trade = TradeLog(
            trade_id=f"buy_{self._trade_counter}_{asset}_{int(time.time())}",
            timestamp=time.time(),
            asset=asset,
            action="buy",
            price=price,
            quantity=quantity,
            pnl=0.0,
        )
        self.trade_log.append(trade)
        self._persist_trade(trade)
        log.debug("paper_portfolio.buy", asset=asset, qty=quantity, price=price, cash=self.cash)
        return trade

    def execute_sell(self, asset: str, price: float, quantity: float) -> TradeLog:
        """Execute a simulated sell order.

        Args:
            asset: Asset symbol.
            price: Current market price.
            quantity: Units to sell.

        Returns:
            The :class:`TradeLog` entry for this trade (includes realised P&L).
        """
        held = self.positions.get(asset, 0.0)
        quantity = min(quantity, held)
        if quantity <= 0:
            self._trade_counter += 1
            trade = TradeLog(
                trade_id=f"hold_{self._trade_counter}",
                timestamp=time.time(),
                asset=asset,
                action="hold",
                price=price,
                quantity=0.0,
            )
            return trade

        proceeds = price * quantity * (1 - self.slippage)
        avg_entry = self.avg_entry_prices.get(asset, price)
        pnl = (price - avg_entry) * quantity

        self.positions[asset] = held - quantity
        if self.positions[asset] <= 0:
            self.positions.pop(asset, None)
            self.avg_entry_prices.pop(asset, None)
        self.cash += proceeds

        self._trade_counter += 1
        trade = TradeLog(
            trade_id=f"sell_{self._trade_counter}_{asset}_{int(time.time())}",
            timestamp=time.time(),
            asset=asset,
            action="sell",
            price=price,
            quantity=quantity,
            pnl=pnl,
        )
        self.trade_log.append(trade)
        self._persist_trade(trade)
        log.debug(
            "paper_portfolio.sell", asset=asset, qty=quantity, price=price, pnl=pnl, cash=self.cash
        )
        return trade

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Compute total portfolio value (cash + positions).

        Args:
            current_prices: Latest mid-prices per asset symbol.

        Returns:
            Total portfolio value in USD.
        """
        pos_value = sum(
            self.positions.get(a, 0.0) * current_prices.get(a, 0.0) for a in self.positions
        )
        return self.cash + pos_value

    def total_pnl(self, current_prices: dict[str, float]) -> float:
        """Total P&L since inception.

        Args:
            current_prices: Latest prices.

        Returns:
            Unrealised + realised P&L in USD.
        """
        return self.total_value(current_prices) - self.starting_cash

    def realised_pnl(self) -> float:
        """Sum of all realised P&L from closed trades."""
        return sum(t.pnl for t in self.trade_log)

    def max_drawdown(self, current_prices: dict[str, float]) -> float:
        """Compute maximum drawdown from peak portfolio value.

        Args:
            current_prices: Latest prices (used for current valuation).

        Returns:
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).
        """
        current = self.total_value(current_prices)
        self._peak_value = max(self._peak_value, current)
        if self._peak_value == 0:
            return 0.0
        return max(0.0, (self._peak_value - current) / self._peak_value)

    def win_rate(self) -> float:
        """Fraction of closed trades with positive P&L."""
        closed = [t for t in self.trade_log if t.action == "sell" and t.pnl != 0]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed)

    def summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Return a comprehensive performance summary dict.

        Args:
            current_prices: Latest market prices.

        Returns:
            Dictionary with total_value, total_pnl_usd, total_pnl_pct,
            realised_pnl, max_drawdown_pct, win_rate, trade_count, positions.
        """
        tv = self.total_value(current_prices)
        return {
            "total_value": round(tv, 2),
            "total_pnl_usd": round(self.total_pnl(current_prices), 2),
            "total_pnl_pct": round(100.0 * (tv - self.starting_cash) / self.starting_cash, 3),
            "realised_pnl": round(self.realised_pnl(), 2),
            "max_drawdown_pct": round(100.0 * self.max_drawdown(current_prices), 3),
            "win_rate": round(self.win_rate(), 4),
            "trade_count": len(self.trade_log),
            "positions": {a: round(q, 6) for a, q in self.positions.items()},
            "cash": round(self.cash, 2),
        }


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------


class PaperTrader:
    """Simulated live trader that connects to Binance WebSocket price feeds.

    Feeds live prices into the LARSA ensemble agents, executes simulated
    trades via :class:`PaperPortfolio`, and optionally runs in mock mode
    for offline testing.

    Args:
        portfolio: The :class:`PaperPortfolio` to trade against.
        assets: List of Binance symbol names (e.g. ``["BTCUSDT", "ETHUSDT"]``).
        agents: Optional list of trained LARSA agents for vote-based decisions.
        decision_fn: Optional callable ``(asset, price) -> action_str`` where
            action_str is ``"buy"``, ``"sell"``, or ``"hold"``.  Takes
            precedence over *agents* when provided.
        trade_qty_fraction: Fraction of cash to deploy per buy signal.
        ws_url: Binance WebSocket base URL.
        mock_prices: When not ``None``, use this dict as a static price
            source (offline/testing mode); ignores the WebSocket entirely.
    """

    def __init__(
        self,
        portfolio: PaperPortfolio,
        assets: list[str] | None = None,
        agents: list[Any] | None = None,
        decision_fn: Callable[[str, float], str] | None = None,
        trade_qty_fraction: float = 0.05,
        ws_url: str | None = None,
        mock_prices: dict[str, list[float]] | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.assets = assets or ["BTCUSDT"]
        self.agents = agents or []
        self.decision_fn = decision_fn
        self.trade_qty_fraction = trade_qty_fraction
        self.ws_url = ws_url or os.environ.get("BINANCE_WS_URL", _DEFAULT_WS_URL)
        self.mock_prices = mock_prices

        # Live price cache
        self._current_prices: dict[str, float] = {a: 0.0 for a in self.assets}
        self._tick_count: int = 0

        log.info(
            "paper_trader.init",
            assets=self.assets,
            mock_mode=mock_prices is not None,
            ws_url=self.ws_url,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(self, asset: str, price: float) -> str:
        """Choose an action for *asset* given current *price*.

        Uses the custom *decision_fn* if provided; otherwise falls back to
        a simple majority vote across the provided agents; or defaults to
        ``"hold"`` if no agents are configured.

        Args:
            asset: Asset symbol.
            price: Current price.

        Returns:
            One of ``"buy"``, ``"sell"``, ``"hold"``.
        """
        if self.decision_fn is not None:
            return self.decision_fn(asset, price)

        if not self.agents:
            return "hold"

        import numpy as np
        import torch as th

        votes: dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        state = th.zeros(12)  # placeholder state; replace with live state builder
        for agent in self.agents:
            try:
                with th.no_grad():
                    q_vals = agent.act(state.unsqueeze(0)).squeeze()
                action_idx = int(q_vals.argmax().item())
                action_str = ["hold", "buy", "sell"][action_idx % 3]
                votes[action_str] += 1
            except Exception as exc:
                log.warning("paper_trader.agent_vote_fail", error=str(exc))

        return max(votes, key=lambda k: votes[k])

    def _execute_decision(self, asset: str, price: float, action: str) -> None:
        """Execute a buy/sell/hold decision.

        Args:
            asset: Asset symbol.
            price: Current price.
            action: ``"buy"``, ``"sell"``, or ``"hold"``.
        """
        if action == "buy":
            qty = (self.portfolio.cash * self.trade_qty_fraction) / max(price, 1e-10)
            if qty > 0:
                self.portfolio.execute_buy(asset, price, qty)
        elif action == "sell":
            qty = self.portfolio.positions.get(asset, 0.0) * 0.5
            if qty > 0:
                self.portfolio.execute_sell(asset, price, qty)
        # "hold" requires no action

    # ------------------------------------------------------------------
    # Mock (offline) run
    # ------------------------------------------------------------------

    def run(self, max_ticks: int = 100) -> None:
        """Run the paper trader synchronously using mock or static price data.

        In mock mode iterates through the provided price lists.
        In live mode raises :class:`NotImplementedError` (use :meth:`run_async`).

        Args:
            max_ticks: Maximum price ticks to process.
        """
        if self.mock_prices is None:
            raise NotImplementedError(
                "Live WebSocket mode requires async; call run_async() or provide mock_prices."
            )

        max_len = max(len(v) for v in self.mock_prices.values()) if self.mock_prices else 0
        max_ticks = min(max_ticks, max_len)

        log.info("paper_trader.run.mock_start", max_ticks=max_ticks)

        for tick in range(max_ticks):
            for asset in self.assets:
                prices_list = self.mock_prices.get(asset, [])
                if tick >= len(prices_list):
                    continue
                price = float(prices_list[tick])
                self._current_prices[asset] = price
                action = self._decide(asset, price)
                self._execute_decision(asset, price, action)

            self._tick_count += 1

            if tick % 50 == 0:
                summary = self.portfolio.summary(self._current_prices)
                log.info(
                    "paper_trader.tick",
                    tick=tick,
                    total_value=summary["total_value"],
                    pnl_pct=summary["total_pnl_pct"],
                )

        log.info("paper_trader.run.done", total_ticks=self._tick_count)

    async def run_async(self, max_ticks: int = 0) -> None:
        """Run the paper trader against the live Binance WebSocket.

        Subscribes to ``<symbol>@miniTicker`` streams for all configured
        assets and processes price ticks until *max_ticks* is reached or
        the task is cancelled.

        Args:
            max_ticks: Stop after this many ticks.  ``0`` means run forever.

        Raises:
            ImportError: If the ``websockets`` package is not installed.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "websockets package required for live mode. "
                "Install with: pip install websockets"
            ) from exc

        streams = "/".join(f"{a.lower()}@miniTicker" for a in self.assets)
        url = f"{self.ws_url}/{streams}"

        log.info("paper_trader.ws.connect", url=url)

        async with websockets.connect(url) as ws:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                    # Support both single and combined stream formats
                    data = msg.get("data", msg)
                    symbol = data.get("s", "")
                    price = float(data.get("c", 0))

                    if symbol in self.assets and price > 0:
                        self._current_prices[symbol] = price
                        action = self._decide(symbol, price)
                        self._execute_decision(symbol, price, action)
                        self._tick_count += 1

                        if self._tick_count % 100 == 0:
                            summary = self.portfolio.summary(self._current_prices)
                            log.info(
                                "paper_trader.live_tick",
                                tick=self._tick_count,
                                total_value=summary["total_value"],
                                pnl_pct=summary["total_pnl_pct"],
                            )

                    if max_ticks > 0 and self._tick_count >= max_ticks:
                        log.info("paper_trader.ws.max_ticks_reached", ticks=self._tick_count)
                        break

                except Exception as exc:
                    log.warning("paper_trader.ws.msg_error", error=str(exc))


# ---------------------------------------------------------------------------
# Comparison: paper vs backtest
# ---------------------------------------------------------------------------


def compare_vs_backtest(
    paper_portfolio: PaperPortfolio,
    backtest_returns: list[float],
    current_prices: dict[str, float],
) -> dict[str, Any]:
    """Compare paper trading performance against a historical backtest.

    Args:
        paper_portfolio: The live paper trading portfolio.
        backtest_returns: List of per-step return fractions from a backtest
            replay (e.g. from :class:`EvalTradeSimulator`).
        current_prices: Current market prices for portfolio valuation.

    Returns:
        Dictionary with side-by-side metrics for paper vs backtest.
    """
    import numpy as np

    paper_summary = paper_portfolio.summary(current_prices)

    bt_arr = np.array(backtest_returns, dtype=float)
    bt_total_return = float(np.prod(1 + bt_arr) - 1) * 100
    bt_sharpe = (
        float(bt_arr.mean() / (bt_arr.std() + 1e-9)) * (252**0.5) if len(bt_arr) > 1 else 0.0
    )
    bt_drawdowns = 1 - bt_arr.cumsum() / (bt_arr.cumsum().cummax() + 1e-9) if len(bt_arr) else []
    bt_max_dd = float(np.array(bt_drawdowns).max()) * 100 if len(bt_drawdowns) else 0.0

    comparison = {
        "paper": {
            "total_pnl_pct": paper_summary["total_pnl_pct"],
            "realised_pnl": paper_summary["realised_pnl"],
            "max_drawdown_pct": paper_summary["max_drawdown_pct"],
            "win_rate": paper_summary["win_rate"],
            "trade_count": paper_summary["trade_count"],
            "total_value": paper_summary["total_value"],
        },
        "backtest": {
            "total_return_pct": round(bt_total_return, 3),
            "sharpe": round(bt_sharpe, 4),
            "max_drawdown_pct": round(bt_max_dd, 3),
            "step_count": len(backtest_returns),
        },
        "delta_return_pct": round(
            paper_summary["total_pnl_pct"] - bt_total_return, 3
        ),
    }

    log.info(
        "compare_vs_backtest",
        paper_pnl=comparison["paper"]["total_pnl_pct"],
        bt_return=comparison["backtest"]["total_return_pct"],
        delta=comparison["delta_return_pct"],
    )
    return comparison


# ---------------------------------------------------------------------------
# Terminal dashboard
# ---------------------------------------------------------------------------


def print_dashboard(
    portfolio: PaperPortfolio,
    current_prices: dict[str, float],
    width: int = 72,
) -> None:
    """Print a simple terminal dashboard of current positions and P&L.

    Args:
        portfolio: The paper trading portfolio.
        current_prices: Current market prices.
        width: Column width for the dashboard.
    """
    sep = "-" * width
    summary = portfolio.summary(current_prices)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(sep)
    print(f"  LARSA Paper Trading Dashboard — {now}")
    print(sep)
    print(f"  Total Value   : ${summary['total_value']:>12,.2f}")
    print(f"  Cash          : ${summary['cash']:>12,.2f}")
    pnl_pct = summary["total_pnl_pct"]
    sign = "+" if pnl_pct >= 0 else ""
    print(f"  Total P&L     : {sign}{pnl_pct:.3f}%  (${summary['total_pnl_usd']:+,.2f})")
    print(f"  Realised P&L  : ${summary['realised_pnl']:>+12,.2f}")
    print(f"  Max Drawdown  : {summary['max_drawdown_pct']:.3f}%")
    print(f"  Win Rate      : {summary['win_rate']*100:.1f}%  ({summary['trade_count']} trades)")
    print(sep)
    print("  POSITIONS")
    if not summary["positions"]:
        print("    (no open positions)")
    else:
        for asset, qty in summary["positions"].items():
            price = current_prices.get(asset, 0.0)
            val = qty * price
            avg = portfolio.avg_entry_prices.get(asset, price)
            unreal_pnl = (price - avg) * qty
            print(
                f"    {asset:<12}  qty={qty:.6f}  "
                f"price=${price:,.2f}  val=${val:,.2f}  "
                f"unreal={unreal_pnl:+,.2f}"
            )
    print(sep)
    print("  RECENT TRADES (last 5)")
    recent = list(reversed(portfolio.trade_log))[:5]
    if not recent:
        print("    (no trades yet)")
    for t in recent:
        ts = datetime.fromtimestamp(t.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        print(
            f"    [{ts}] {t.action.upper():<4}  {t.asset:<12}  "
            f"qty={t.quantity:.6f}  @${t.price:,.2f}  "
            f"pnl={t.pnl:+,.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Quick offline smoke test."""
    import numpy as np

    # Generate mock BTC price series with a gentle uptrend
    rng = np.random.default_rng(42)
    prices = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.002, 200)))

    portfolio = PaperPortfolio(starting_cash=100_000.0, db_path=":memory:")

    # Simple alternating buy/sell strategy for testing
    def simple_strategy(asset: str, price: float) -> str:
        tick = _smoke_test.__dict__.setdefault("_tick", 0)
        _smoke_test.__dict__["_tick"] += 1
        if tick % 10 == 0:
            return "buy"
        if tick % 10 == 5:
            return "sell"
        return "hold"

    trader = PaperTrader(
        portfolio=portfolio,
        assets=["BTCUSDT"],
        decision_fn=simple_strategy,
        mock_prices={"BTCUSDT": prices.tolist()},
    )
    trader.run(max_ticks=100)

    current = {"BTCUSDT": float(prices[-1])}
    print_dashboard(portfolio, current)

    comparison = compare_vs_backtest(
        portfolio,
        backtest_returns=list(np.diff(prices) / prices[:-1])[:100],
        current_prices=current,
    )
    log.info("smoke_test.comparison", result=comparison)


if __name__ == "__main__":
    _smoke_test()

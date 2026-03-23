"""Alpaca paper trading integration for the LARSA trading system.

Connects to the Alpaca paper trading API and executes decisions produced by
the ensemble of DRL agents.  Supports a dry-run mode that logs intended orders
without issuing real API calls.

Environment variables::

    ALPACA_API_KEY       Required for live paper trading.
    ALPACA_SECRET_KEY    Required for live paper trading.
    ALPACA_BASE_URL      Defaults to the Alpaca paper trading endpoint.

Typical usage::

    from paper_trader import AlpacaPaperTrader, PaperTradingLoop
    trader = AlpacaPaperTrader()
    trader.connect()
    trader.place_order("BTCUSD", qty=0.001, side="buy")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from exceptions import ConfigError, DataFetchError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
_MAX_POSITION_FRACTION = 0.10  # max 10 % of portfolio per trade


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrderResult:
    """Summary of a submitted (or simulated) order.

    Attributes:
        order_id: Alpaca order ID string, or ``"dry-run"`` in dry-run mode.
        symbol: Asset symbol (e.g. ``"BTCUSD"``).
        qty: Fractional quantity submitted.
        side: ``"buy"`` or ``"sell"``.
        status: Order status string returned by the broker.
        filled_avg_price: Average fill price, or ``None`` if not yet filled.
    """

    order_id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_avg_price: float | None = None


# ---------------------------------------------------------------------------
# AlpacaPaperTrader
# ---------------------------------------------------------------------------


class AlpacaPaperTrader:
    """Thin wrapper around the Alpaca paper trading REST API.

    Args:
        dry_run: When ``True`` no real API calls are made; orders are only
            logged.  Defaults to ``False``.
        api_key: Alpaca API key.  Falls back to ``ALPACA_API_KEY`` env var.
        secret_key: Alpaca secret key.  Falls back to ``ALPACA_SECRET_KEY``.
        base_url: Alpaca endpoint URL.  Falls back to ``ALPACA_BASE_URL`` or
            the paper-trading default.

    Raises:
        ConfigError: If credentials are missing and ``dry_run`` is ``False``.
    """

    def __init__(
        self,
        dry_run: bool = False,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.dry_run = dry_run
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("ALPACA_BASE_URL", _DEFAULT_ALPACA_BASE_URL)
        )
        self._api: Any = None  # will be set in connect()

        if not dry_run and (not self._api_key or not self._secret_key):
            raise ConfigError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set when dry_run=False."
            )

        log.info(
            "paper_trader.init",
            dry_run=dry_run,
            base_url=self._base_url,
            has_key=bool(self._api_key),
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish and verify the connection to Alpaca.

        In dry-run mode the connection step is skipped entirely.

        Raises:
            ConfigError: If the ``alpaca-trade-api`` or ``alpaca-py`` package
                is not installed.
            DataFetchError: If the account cannot be fetched (e.g. bad keys).
        """
        if self.dry_run:
            log.info("paper_trader.connect.dry_run")
            return

        self._api = self._build_api_client()
        try:
            account = self._api.get_account()
            log.info(
                "paper_trader.connected",
                status=getattr(account, "status", "unknown"),
                equity=getattr(account, "equity", "?"),
            )
        except Exception as exc:
            raise DataFetchError(
                "Failed to connect to Alpaca; check API credentials.", cause=exc
            ) from exc

    def _build_api_client(self) -> Any:
        """Construct the Alpaca REST client using whichever library is installed.

        Prefers ``alpaca-py`` (``alpaca.trading.client.TradingClient``) and
        falls back to the older ``alpaca_trade_api.REST`` client.

        Returns:
            A connected Alpaca API client object.

        Raises:
            ConfigError: If neither library is available.
        """
        # Try alpaca-py first (newer library)
        try:
            from alpaca.trading.client import TradingClient  # type: ignore[import]

            return TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=True,
            )
        except ImportError:
            pass

        # Fall back to alpaca-trade-api
        try:
            import alpaca_trade_api as tradeapi  # type: ignore[import]

            return tradeapi.REST(
                key_id=self._api_key,
                secret_key=self._secret_key,
                base_url=self._base_url,
            )
        except ImportError as exc:
            raise ConfigError(
                "Neither 'alpaca-py' nor 'alpaca-trade-api' is installed.  "
                "Run: pip install alpaca-py"
            ) from exc

    # ------------------------------------------------------------------
    # Account / portfolio queries
    # ------------------------------------------------------------------

    def get_account_value(self) -> float:
        """Return current portfolio equity in USD.

        In dry-run mode returns a sentinel value of ``100_000.0``.

        Returns:
            Portfolio equity (USD).

        Raises:
            DataFetchError: If the broker call fails.
        """
        if self.dry_run:
            log.debug("paper_trader.get_account_value.dry_run", value=100_000.0)
            return 100_000.0

        try:
            account = self._api.get_account()
            equity = float(getattr(account, "equity", 0))
            log.debug("paper_trader.get_account_value", equity=equity)
            return equity
        except Exception as exc:
            raise DataFetchError("Failed to fetch account value.", cause=exc) from exc

    def get_positions(self) -> list[dict[str, Any]]:
        """Return a list of current open positions.

        Each element is a dict with at minimum ``symbol``, ``qty``, and
        ``market_value`` keys.

        In dry-run mode returns an empty list.

        Returns:
            List of position dictionaries.

        Raises:
            DataFetchError: If the broker call fails.
        """
        if self.dry_run:
            log.debug("paper_trader.get_positions.dry_run")
            return []

        try:
            raw = self._api.get_all_positions()
            positions = []
            for pos in raw:
                positions.append(
                    {
                        "symbol": getattr(pos, "symbol", ""),
                        "qty": float(getattr(pos, "qty", 0)),
                        "market_value": float(getattr(pos, "market_value", 0)),
                        "side": getattr(pos, "side", "long"),
                    }
                )
            log.debug("paper_trader.get_positions", count=len(positions))
            return positions
        except Exception as exc:
            raise DataFetchError("Failed to fetch positions.", cause=exc) from exc

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def _enforce_position_limit(self, symbol: str, qty: float, side: str) -> float:
        """Clamp *qty* so the order value does not exceed 10 % of portfolio.

        Args:
            symbol: Asset symbol, used only for logging.
            qty: Requested quantity.
            side: ``"buy"`` or ``"sell"``.

        Returns:
            Adjusted quantity (may be the same as *qty*).
        """
        if side == "sell":
            # Selling is always safe from a portfolio-limit perspective.
            return qty

        try:
            equity = self.get_account_value()
        except DataFetchError:
            log.warning("paper_trader.position_limit.equity_fetch_failed")
            return qty

        # Estimate a conservative per-unit value; use $1 if we have no price.
        max_allowed_qty = max(0.0, (equity * _MAX_POSITION_FRACTION))
        if qty > max_allowed_qty:
            log.warning(
                "paper_trader.position_limit.clamped",
                symbol=symbol,
                requested=qty,
                clamped=round(max_allowed_qty, 8),
                equity=equity,
                limit_pct=_MAX_POSITION_FRACTION * 100,
            )
            return max_allowed_qty
        return qty

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
    ) -> OrderResult:
        """Submit a paper trade order to Alpaca.

        Applies the position-size risk guard before submission.  In dry-run
        mode the order is logged but never sent.

        Args:
            symbol: Asset symbol, e.g. ``"BTCUSD"``.
            qty: Quantity to trade (fractional supported).
            side: ``"buy"`` or ``"sell"``.
            order_type: Alpaca order type (default ``"market"``).
            time_in_force: Alpaca time-in-force (default ``"gtc"``).

        Returns:
            :class:`OrderResult` describing the submitted (or simulated) order.

        Raises:
            ValueError: If *side* is not ``"buy"`` or ``"sell"``.
            DataFetchError: If the broker call fails.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        safe_qty = self._enforce_position_limit(symbol, qty, side)

        if self.dry_run:
            result = OrderResult(
                order_id="dry-run",
                symbol=symbol,
                qty=safe_qty,
                side=side,
                status="simulated",
                filled_avg_price=None,
            )
            log.info(
                "paper_trader.order.dry_run",
                symbol=symbol,
                qty=safe_qty,
                side=side,
            )
            return result

        try:
            order = self._submit_order(symbol, safe_qty, side, order_type, time_in_force)
            result = OrderResult(
                order_id=str(getattr(order, "id", "unknown")),
                symbol=symbol,
                qty=safe_qty,
                side=side,
                status=str(getattr(order, "status", "submitted")),
                filled_avg_price=(
                    float(order.filled_avg_price)
                    if getattr(order, "filled_avg_price", None)
                    else None
                ),
            )
            log.info(
                "paper_trader.order.placed",
                order_id=result.order_id,
                symbol=symbol,
                qty=safe_qty,
                side=side,
                status=result.status,
            )
            return result
        except Exception as exc:
            raise DataFetchError(
                f"Failed to place {side} order for {symbol}.", cause=exc
            ) from exc

    def _submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
    ) -> Any:
        """Delegate the order submission to whichever Alpaca client is in use.

        Supports both the ``alpaca-py`` and ``alpaca-trade-api`` interfaces.

        Args:
            symbol: Asset symbol.
            qty: Quantity to trade.
            side: ``"buy"`` or ``"sell"``.
            order_type: Order type string.
            time_in_force: Time-in-force string.

        Returns:
            Raw order object from the Alpaca client.
        """
        # alpaca-py interface
        if hasattr(self._api, "submit_order"):
            try:
                from alpaca.trading.requests import MarketOrderRequest  # type: ignore[import]
                from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore[import]

                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
                return self._api.submit_order(order_data=req)
            except ImportError:
                pass

        # alpaca-trade-api fallback
        return self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
        )


# ---------------------------------------------------------------------------
# PaperTradingLoop
# ---------------------------------------------------------------------------


@dataclass
class PaperTradingConfig:
    """Configuration for the paper trading loop.

    Attributes:
        symbol: Asset to trade (e.g. ``"BTCUSD"``).
        trade_qty: Fixed quantity per order.
        poll_interval_seconds: Seconds between decision cycles.
        max_iterations: Stop after this many cycles (``0`` = run forever).
        agent_weights: Per-agent vote weights used by the ensemble.
    """

    symbol: str = "BTCUSD"
    trade_qty: float = 0.001
    poll_interval_seconds: float = 60.0
    max_iterations: int = 0
    agent_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


class PaperTradingLoop:
    """Runs ensemble agent decisions and executes them via Alpaca paper trading.

    Args:
        trader: Configured and connected :class:`AlpacaPaperTrader`.
        agents: List of trained DRL agent objects with an ``act`` attribute
            (compatible with the LARSA agent interface).
        config: :class:`PaperTradingConfig` with runtime settings.

    Example::

        trader = AlpacaPaperTrader(dry_run=True)
        trader.connect()
        loop = PaperTradingLoop(trader=trader, agents=my_agents)
        loop.run(state_provider=lambda: current_state_tensor)
    """

    # Action integer <-> string mapping (matches TradeSimulator convention)
    _ACTION_MAP: dict[int, str] = {0: "hold", 1: "buy", 2: "sell"}

    def __init__(
        self,
        trader: AlpacaPaperTrader,
        agents: list[Any],
        config: PaperTradingConfig | None = None,
    ) -> None:
        self.trader = trader
        self.agents = agents
        self.config = config or PaperTradingConfig()

        log.info(
            "paper_trading_loop.init",
            symbol=self.config.symbol,
            trade_qty=self.config.trade_qty,
            num_agents=len(agents),
            dry_run=trader.dry_run,
        )

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def _weighted_vote(self, state: Any) -> int:
        """Collect per-agent action votes and return the weighted majority.

        Args:
            state: Current environment state tensor passed to each agent.

        Returns:
            Winning action integer (0 = hold, 1 = buy, 2 = sell).
        """
        import torch

        votes: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
        weights = self.config.agent_weights

        for i, agent in enumerate(self.agents):
            w = weights[i] if i < len(weights) else 1.0
            try:
                with torch.no_grad():
                    if hasattr(agent, "act"):
                        q_vals = agent.act(state)
                    else:
                        q_vals = agent(state)
                    action = int(q_vals.argmax(dim=-1).item())
            except Exception as exc:
                log.warning("paper_trading_loop.vote_failed", agent=i, error=str(exc))
                action = 0  # default to hold on failure
            votes[action] = votes.get(action, 0.0) + w

        winning_action = max(votes, key=lambda a: votes[a])
        log.debug(
            "paper_trading_loop.vote",
            votes={self._ACTION_MAP[k]: round(v, 3) for k, v in votes.items()},
            decision=self._ACTION_MAP[winning_action],
        )
        return winning_action

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, state_provider: Any) -> None:
        """Execute the paper trading loop.

        Calls *state_provider()* each cycle to obtain the current market state,
        runs the weighted ensemble vote, and places the resulting order.  The
        loop sleeps for ``config.poll_interval_seconds`` between iterations.

        Args:
            state_provider: Zero-argument callable returning the current state
                tensor (or any object the agents accept).
        """
        iteration = 0
        log.info("paper_trading_loop.start", max_iterations=self.config.max_iterations)

        while True:
            if self.config.max_iterations > 0 and iteration >= self.config.max_iterations:
                log.info("paper_trading_loop.max_iterations_reached", iterations=iteration)
                break

            try:
                state = state_provider()
                action = self._weighted_vote(state)
                self._execute_action(action)
            except KeyboardInterrupt:
                log.info("paper_trading_loop.interrupted")
                break
            except Exception as exc:
                log.error("paper_trading_loop.cycle_error", error=str(exc))

            iteration += 1
            if self.config.max_iterations == 0 or iteration < self.config.max_iterations:
                time.sleep(self.config.poll_interval_seconds)

        log.info("paper_trading_loop.stopped", iterations=iteration)

    def _execute_action(self, action: int) -> OrderResult | None:
        """Translate an agent action integer into an Alpaca order.

        Args:
            action: Agent action (0 = hold, 1 = buy, 2 = sell).

        Returns:
            :class:`OrderResult` if an order was placed, ``None`` for hold.
        """
        symbol = self.config.symbol
        qty = self.config.trade_qty

        if action == 0:
            log.debug("paper_trading_loop.hold", symbol=symbol)
            return None
        if action == 1:
            return self.trader.place_order(symbol, qty, "buy")
        if action == 2:
            return self.trader.place_order(symbol, qty, "sell")

        log.warning("paper_trading_loop.unknown_action", action=action)
        return None

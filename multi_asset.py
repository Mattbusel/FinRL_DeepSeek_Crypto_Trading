"""Multi-asset portfolio extension for the LARSA trading system.

Extends the single-asset :class:`TradeSimulator` to support portfolios of N
crypto assets simultaneously.  Key components:

* :class:`PortfolioState` -- snapshot of prices, positions, cash, and total value.
* :class:`PortfolioAction` -- target allocation weights that sum to 1.0.
* :class:`CorrelationMatrix` -- rolling Pearson correlation tracker used as
  additional state features.
* :class:`MultiAssetEnvironment` -- vectorised environment that wraps N single-
  asset price series and exposes a combined observation / action space.
* :class:`MultiAssetEnsemble` -- coordinates per-asset agents plus a portfolio-
  level meta-agent that optimises allocation weights.

Example::

    from multi_asset import MultiAssetEnvironment, MultiAssetEnsemble
    from erl_agent import AgentD3QN

    env = MultiAssetEnvironment(
        assets=["BTC", "ETH", "SOL"],
        starting_cash=100_000.0,
    )
    state = env.reset()
    print(state.shape)  # (state_dim,)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch as th

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a single time step.

    Attributes:
        prices: Current mid-price per asset symbol.
        positions: Number of units held per asset (can be fractional).
        cash: Remaining cash balance in USD.
        total_value: Cash + market value of all positions.
        step: Current environment step index.
    """

    prices: dict[str, float]
    positions: dict[str, float]
    cash: float
    total_value: float
    step: int = 0

    def position_value(self) -> dict[str, float]:
        """Return the USD value of each position."""
        return {asset: self.positions[asset] * self.prices[asset] for asset in self.prices}

    def weights(self) -> dict[str, float]:
        """Return current portfolio weights (fraction of total value)."""
        if self.total_value == 0:
            return {asset: 0.0 for asset in self.prices}
        pv = self.position_value()
        return {asset: pv[asset] / self.total_value for asset in self.prices}


@dataclass
class PortfolioAction:
    """Target allocation weights across all assets.

    Attributes:
        weights: Target fraction of portfolio value per asset.  Must sum to
            a value <= 1.0 (the remainder is held as cash).
        rebalance_threshold: Only trade if current weight deviates by more
            than this fraction (avoids micro-trades on tiny deviations).
    """

    weights: dict[str, float]
    rebalance_threshold: float = 0.02

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"Portfolio weights must sum to <= 1.0, got {total:.4f}. "
                "The remainder is implicitly held as cash."
            )
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("All portfolio weights must be non-negative.")

    def normalize(self) -> "PortfolioAction":
        """Return a copy with weights normalised to sum exactly to 1.0."""
        total = sum(self.weights.values())
        if total == 0:
            return PortfolioAction(
                weights={k: 1.0 / len(self.weights) for k in self.weights},
                rebalance_threshold=self.rebalance_threshold,
            )
        return PortfolioAction(
            weights={k: v / total for k, v in self.weights.items()},
            rebalance_threshold=self.rebalance_threshold,
        )


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


class CorrelationMatrix:
    """Rolling Pearson correlation tracker for a basket of assets.

    Maintains a sliding window of returns and computes the pairwise
    correlation matrix on demand.  The flattened upper-triangle of the matrix
    is exposed as additional state features for downstream agents.

    Args:
        assets: Ordered list of asset symbols.
        window: Number of return observations in the rolling window.

    Example::

        cm = CorrelationMatrix(["BTC", "ETH", "SOL"], window=60)
        cm.update({"BTC": 30000.0, "ETH": 2000.0, "SOL": 50.0})
        features = cm.feature_vector()  # shape (3,) for 3 assets
    """

    def __init__(self, assets: list[str], window: int = 120) -> None:
        self.assets = assets
        self.window = window
        self._price_history: dict[str, deque[float]] = {
            asset: deque(maxlen=window + 1) for asset in assets
        }

    def update(self, prices: dict[str, float]) -> None:
        """Append a new price observation for each asset.

        Args:
            prices: Mapping of asset symbol to current mid-price.
        """
        for asset in self.assets:
            if asset in prices:
                self._price_history[asset].append(prices[asset])

    def correlation_matrix(self) -> np.ndarray:
        """Compute the current rolling Pearson correlation matrix.

        Returns:
            Square matrix of shape ``(N, N)`` where ``N = len(assets)``.
            Returns the identity matrix if insufficient history is available.
        """
        n = len(self.assets)
        returns: list[np.ndarray] = []
        for asset in self.assets:
            hist = list(self._price_history[asset])
            if len(hist) < 2:
                return np.eye(n)
            arr = np.array(hist, dtype=float)
            ret = np.diff(arr) / (arr[:-1] + 1e-10)
            returns.append(ret)

        min_len = min(len(r) for r in returns)
        if min_len < 2:
            return np.eye(n)

        mat = np.stack([r[-min_len:] for r in returns])  # (N, T)
        # Pearson correlation across assets
        try:
            corr = np.corrcoef(mat)
        except Exception:
            corr = np.eye(n)
        # Replace NaNs with 0
        corr = np.nan_to_num(corr, nan=0.0)
        return corr

    def feature_vector(self) -> np.ndarray:
        """Return the flattened upper-triangle of the correlation matrix.

        For N assets this produces N*(N-1)//2 features.

        Returns:
            1-D array of correlation values suitable for concatenation into a
            state vector.
        """
        corr = self.correlation_matrix()
        n = corr.shape[0]
        idx = np.triu_indices(n, k=1)
        return corr[idx]

    @property
    def feature_dim(self) -> int:
        """Number of features produced by :meth:`feature_vector`."""
        n = len(self.assets)
        return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# MultiAssetEnvironment
# ---------------------------------------------------------------------------


class MultiAssetEnvironment:
    """Vectorised multi-asset portfolio trading environment.

    Each step the agent provides target allocation weights; the environment
    rebalances positions, applies slippage, and returns a combined state
    vector that includes per-asset price features, current portfolio weights,
    and rolling inter-asset correlations.

    Args:
        assets: Ordered list of asset symbols (e.g. ``["BTC", "ETH", "SOL"]``).
        price_data: Optional mapping of asset symbol to 1-D numpy price array.
            If not provided, synthetic random-walk data is generated for
            testing purposes.
        starting_cash: Initial portfolio cash balance (USD).
        slippage: Per-trade slippage fraction applied to each rebalancing trade.
        correlation_window: Rolling window length for the correlation matrix.
        max_step: Maximum number of steps before the episode terminates.
        device: PyTorch device for state tensors.

    Environment metadata (consumed by ElegantRL-style training loops):

    * ``env_name`` -- ``"MultiAsset-v0"``
    * ``state_dim`` -- total dimension of the state vector
    * ``action_dim`` -- ``len(assets)`` (one weight per asset)
    * ``if_discrete`` -- ``False`` (continuous allocation weights)
    * ``target_return`` -- ``+inf``
    """

    env_name: str = "MultiAsset-v0"
    if_discrete: bool = False
    target_return: float = float("inf")

    def __init__(
        self,
        assets: list[str] | None = None,
        price_data: dict[str, np.ndarray] | None = None,
        starting_cash: float = 100_000.0,
        slippage: float = 5e-4,
        correlation_window: int = 120,
        max_step: int = 1800,
        device: th.device | None = None,
    ) -> None:
        self.assets: list[str] = assets or ["BTC", "ETH", "SOL"]
        self.n_assets: int = len(self.assets)
        self.starting_cash = starting_cash
        self.slippage = slippage
        self.max_step = max_step
        self.device = device or th.device("cpu")

        # Price series: dict[asset] -> np.ndarray shape (T,)
        if price_data is not None:
            self._price_data = price_data
        else:
            log.warning(
                "multi_asset.env.synthetic_prices",
                reason="No price_data provided; using synthetic random-walk data.",
            )
            self._price_data = self._generate_synthetic(max_step * 2)

        self._min_len: int = min(len(v) for v in self._price_data.values())

        # Correlation tracker
        self.correlation = CorrelationMatrix(self.assets, window=correlation_window)

        # State dimension:
        #   per-asset features: 2 (normalised price change, current weight) * N
        #   portfolio-level:    2 (cash fraction, step fraction)
        #   correlation upper-triangle: N*(N-1)//2
        per_asset_dim = 2 * self.n_assets
        portfolio_dim = 2
        corr_dim = self.correlation.feature_dim
        self.state_dim: int = per_asset_dim + portfolio_dim + corr_dim

        # Action dim: one weight per asset (continuous [0, 1])
        self.action_dim: int = self.n_assets

        # Episode state (reset in reset())
        self._step_i: int = 0
        self._offset: int = 0
        self._prices: dict[str, float] = {a: 0.0 for a in self.assets}
        self._prev_prices: dict[str, float] = {a: 0.0 for a in self.assets}
        self._positions: dict[str, float] = {a: 0.0 for a in self.assets}
        self._cash: float = starting_cash
        self._portfolio_value: float = starting_cash

        log.info(
            "multi_asset.env.init",
            assets=self.assets,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_step=max_step,
        )

    # ------------------------------------------------------------------
    # Synthetic data helper
    # ------------------------------------------------------------------

    def _generate_synthetic(self, length: int) -> dict[str, np.ndarray]:
        """Generate correlated geometric Brownian motion price series.

        Args:
            length: Number of time steps to generate.

        Returns:
            Mapping of asset symbol to price array.
        """
        rng = np.random.default_rng(seed=42)
        base_prices = {"BTC": 30_000.0, "ETH": 2_000.0, "SOL": 50.0}
        result: dict[str, np.ndarray] = {}

        # Shared market factor for mild correlation
        market = rng.normal(0, 0.001, size=length)

        for i, asset in enumerate(self.assets):
            base = base_prices.get(asset, 100.0 * (i + 1))
            idio = rng.normal(0, 0.002, size=length)
            returns = 0.4 * market + 0.6 * idio
            prices = base * np.exp(np.cumsum(returns))
            result[asset] = prices.astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def _get_prices(self, idx: int) -> dict[str, float]:
        """Extract prices at index *idx* for all assets."""
        return {
            asset: float(self._price_data[asset][idx])
            for asset in self.assets
            if idx < len(self._price_data[asset])
        }

    def _portfolio_value_at(self, prices: dict[str, float]) -> float:
        """Compute total portfolio value given *prices*."""
        pv = sum(self._positions[a] * prices.get(a, 0.0) for a in self.assets)
        return self._cash + pv

    def _rebalance(self, target_weights: dict[str, float], prices: dict[str, float]) -> float:
        """Execute trades to reach *target_weights* and return total slippage cost.

        Args:
            target_weights: Target asset weight per symbol (values in [0, 1]).
            prices: Current mid-prices.

        Returns:
            Total slippage cost incurred by this rebalance.
        """
        total_value = self._portfolio_value_at(prices)
        total_slippage = 0.0

        for asset in self.assets:
            target_val = total_value * target_weights.get(asset, 0.0)
            current_val = self._positions[asset] * prices.get(asset, 1.0)
            delta_val = target_val - current_val

            if abs(delta_val) < 1.0:  # ignore sub-dollar trades
                continue

            price = prices.get(asset, 1.0)
            if price <= 0:
                continue

            qty_delta = delta_val / price
            slip = abs(delta_val) * self.slippage
            total_slippage += slip

            self._positions[asset] += qty_delta
            self._cash -= delta_val + (slip if delta_val > 0 else -slip)

        return total_slippage

    # ------------------------------------------------------------------
    # gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state.

        Returns:
            Initial state vector of shape ``(state_dim,)`` as ``float32``.
        """
        self._step_i = 0
        # Random episode start offset
        max_offset = max(0, self._min_len - self.max_step - 10)
        self._offset = int(np.random.randint(0, max(1, max_offset)))

        self._prices = self._get_prices(self._offset)
        self._prev_prices = dict(self._prices)
        self._positions = {a: 0.0 for a in self.assets}
        self._cash = self.starting_cash
        self._portfolio_value = self.starting_cash

        # Seed correlation history with a burn-in window
        self.correlation = CorrelationMatrix(self.assets, window=self.correlation.window)
        for burn_idx in range(max(0, self._offset - 60), self._offset + 1):
            self.correlation.update(self._get_prices(burn_idx))

        log.debug("multi_asset.env.reset", offset=self._offset)
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: 1-D array of shape ``(n_assets,)`` with raw allocation
                weights (will be softmax-normalised to sum to 1.0).

        Returns:
            Tuple of ``(state, reward, done, info)`` where ``state`` is
            ``float32`` of shape ``(state_dim,)``, ``reward`` is a scalar
            float, and ``done`` is a bool.
        """
        self._step_i += 1
        idx = self._offset + self._step_i

        # Advance prices
        self._prev_prices = dict(self._prices)
        self._prices = self._get_prices(min(idx, self._min_len - 1))
        self.correlation.update(self._prices)

        # Normalise action via softmax -> target weights
        action_arr = np.asarray(action, dtype=np.float32)
        exp_a = np.exp(action_arr - action_arr.max())
        weights_arr = exp_a / (exp_a.sum() + 1e-10)
        target_weights = {asset: float(weights_arr[i]) for i, asset in enumerate(self.assets)}

        # Rebalance positions
        old_value = self._portfolio_value
        self._rebalance(target_weights, self._prices)
        self._portfolio_value = self._portfolio_value_at(self._prices)

        reward = float(self._portfolio_value - old_value)
        done = self._step_i >= self.max_step

        info: dict[str, Any] = {
            "portfolio_value": self._portfolio_value,
            "step": self._step_i,
            "weights": target_weights,
        }

        state = self._get_state()
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Build the current observation vector.

        Returns:
            Float32 array of shape ``(state_dim,)``.
        """
        parts: list[np.ndarray] = []

        # Per-asset: (normalised price return, current portfolio weight)
        pv = self._portfolio_value if self._portfolio_value > 0 else 1.0
        for asset in self.assets:
            prev = self._prev_prices.get(asset, self._prices.get(asset, 1.0))
            curr = self._prices.get(asset, prev)
            ret = (curr - prev) / (prev + 1e-10)
            weight = (self._positions[asset] * curr) / pv
            parts.append(np.array([ret, weight], dtype=np.float32))

        # Portfolio-level: (cash fraction, step progress)
        cash_frac = min(self._cash / (pv + 1e-10), 1.0)
        step_frac = self._step_i / max(self.max_step, 1)
        parts.append(np.array([cash_frac, step_frac], dtype=np.float32))

        # Correlation features
        parts.append(self.correlation.feature_vector().astype(np.float32))

        return np.concatenate(parts)

    def get_portfolio_state(self) -> PortfolioState:
        """Return a :class:`PortfolioState` snapshot of the current episode."""
        return PortfolioState(
            prices=dict(self._prices),
            positions=dict(self._positions),
            cash=self._cash,
            total_value=self._portfolio_value,
            step=self._step_i,
        )


# ---------------------------------------------------------------------------
# MultiAssetEnsemble
# ---------------------------------------------------------------------------


class MultiAssetEnsemble:
    """Coordinates per-asset agents and a portfolio-level meta-agent.

    The meta-agent receives the concatenated Q-value vectors from all per-asset
    agents plus the correlation feature vector, and outputs final allocation
    weights via a softmax over a learned linear combination.

    Args:
        asset_agents: Mapping of asset symbol to a trained LARSA-compatible
            agent (must expose an ``act(state)`` method returning Q-values).
        n_assets: Total number of assets (used for action sizing).
        device: Torch device.

    Example::

        ensemble = MultiAssetEnsemble(
            asset_agents={"BTC": btc_agent, "ETH": eth_agent},
            n_assets=2,
        )
        weights = ensemble.get_allocation(env.get_portfolio_state(), correlation)
    """

    def __init__(
        self,
        asset_agents: dict[str, Any],
        n_assets: int,
        device: th.device | None = None,
    ) -> None:
        self.asset_agents = asset_agents
        self.n_assets = n_assets
        self.device = device or th.device("cpu")
        # Simple learnable meta-weight per asset (initialised uniformly)
        self._meta_weights: np.ndarray = np.ones(n_assets) / n_assets

        log.info(
            "multi_asset.ensemble.init",
            assets=list(asset_agents.keys()),
            n_assets=n_assets,
        )

    def _agent_q_values(self, asset: str, state: np.ndarray) -> np.ndarray:
        """Query an individual asset agent for its Q-values.

        Args:
            asset: Asset symbol.
            state: State vector for that asset's sub-environment.

        Returns:
            Q-value array (shape depends on agent action_dim).
        """
        agent = self.asset_agents.get(asset)
        if agent is None:
            return np.zeros(3)
        try:
            state_t = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.no_grad():
                q = agent.act(state_t)
            return q.cpu().numpy().flatten()
        except Exception as exc:
            log.warning("multi_asset.ensemble.q_fail", asset=asset, error=str(exc))
            return np.zeros(3)

    def get_allocation(
        self,
        portfolio_state: PortfolioState,
        correlation: CorrelationMatrix,
        per_asset_states: dict[str, np.ndarray] | None = None,
    ) -> PortfolioAction:
        """Compute target allocation weights from per-asset Q-values.

        The meta-policy takes the argmax Q-value for each asset as a directional
        score, scales by the learned meta-weights, and normalises via softmax.

        Args:
            portfolio_state: Current portfolio snapshot.
            correlation: Live correlation matrix tracker.
            per_asset_states: Optional mapping of asset to state vector.  When
                omitted a zero-vector is used for each asset.

        Returns:
            :class:`PortfolioAction` with normalised weights.
        """
        directional_scores: list[float] = []

        for i, asset in enumerate(self.asset_agents):
            state = (per_asset_states or {}).get(asset, np.zeros(12, dtype=np.float32))
            q_vals = self._agent_q_values(asset, state)
            # Map argmax action to directional score: buy=+1, hold=0, sell=-1
            best_action = int(np.argmax(q_vals))
            direction = [0.0, 1.0, -1.0][best_action % 3]
            # Weight by correlation-adjusted meta-weight
            score = max(0.0, direction) * self._meta_weights[i]
            directional_scores.append(score)

        scores_arr = np.array(directional_scores, dtype=float)
        total = scores_arr.sum()
        if total < 1e-9:
            norm_weights = np.ones(self.n_assets) / self.n_assets
        else:
            norm_weights = scores_arr / total

        assets = list(self.asset_agents.keys())
        return PortfolioAction(
            weights={assets[i]: float(norm_weights[i]) for i in range(len(assets))}
        )

    def update_meta_weights(self, performance: dict[str, float]) -> None:
        """Update meta-weights proportionally to recent per-asset Sharpe ratios.

        Args:
            performance: Mapping of asset symbol to its recent Sharpe ratio (or
                any positive scalar performance metric).
        """
        assets = list(self.asset_agents.keys())
        scores = np.array([max(0.0, performance.get(a, 0.0)) for a in assets])
        total = scores.sum()
        if total > 0:
            self._meta_weights = scores / total
        log.debug("multi_asset.ensemble.meta_weights", weights=dict(zip(assets, self._meta_weights)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Quick smoke test for the multi-asset environment and ensemble."""
    env = MultiAssetEnvironment(
        assets=["BTC", "ETH", "SOL"],
        starting_cash=100_000.0,
        max_step=50,
    )
    state = env.reset()
    assert state.shape == (env.state_dim,), f"Bad state shape: {state.shape}"

    total_reward = 0.0
    done = False
    for _ in range(60):
        if done:
            state = env.reset()
        action = np.random.randn(env.n_assets).astype(np.float32)
        state, reward, done, info = env.step(action)
        total_reward += reward

    log.info("smoke_test.done", state_dim=env.state_dim, total_reward=round(total_reward, 2))
    pstate = env.get_portfolio_state()
    log.info(
        "smoke_test.portfolio",
        total_value=round(pstate.total_value, 2),
        cash=round(pstate.cash, 2),
    )

    # Correlation matrix
    corr = env.correlation
    feat = corr.feature_vector()
    log.info("smoke_test.corr_features", shape=feat.shape, sample=feat[:3].tolist())


if __name__ == "__main__":
    _smoke_test()

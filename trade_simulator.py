"""Vectorised BTC trade simulation environments for the LARSA system.

Two environment variants are provided:

* :class:`TradeSimulator` -- training environment with random episode starts.
* :class:`EvalTradeSimulator` -- evaluation environment with deterministic
  (sequential) episode starts and a tighter stop-loss threshold.

Both inherit from :class:`TradeSimulator` and expose the standard
``reset`` / ``step`` interface consumed by the ElegantRL training loop.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch as th

from data_config import ConfigData
from logger import get_logger

logger = get_logger(__name__)


class TradeSimulator:
    """Vectorised Bitcoin trading simulation with multiple parallel episodes.

    Loads price data and RNN-derived factor predictions from disk at
    construction time.  Each call to :meth:`reset` starts ``num_sims``
    independent episodes at random offsets within the price series.

    Args:
        num_sims: Number of parallel simulation environments.
        slippage: Per-trade slippage as a fraction of the mid price.
        max_position: Maximum absolute BTC position the agent may hold.
        step_gap: Number of 1-second ticks aggregated into one step.
        delay_step: Action execution delay in steps (unused in current logic
            but stored for reference).
        num_ignore_step: Initial steps to skip at episode start.
        device: PyTorch device to place state tensors on.
        gpu_id: GPU index; if >= 0 overrides *device* with the CUDA device.

    Raises:
        FileNotFoundError: If the factor NPY or price CSV files are missing.
        AssertionError: If price and factor array lengths do not match after
            alignment.
    """

    def __init__(
        self,
        num_sims: int = 64,
        slippage: float = 5e-5,
        max_position: int = 2,
        step_gap: int = 1,
        delay_step: int = 1,
        num_ignore_step: int = 60,
        device: th.device = th.device("cpu"),
        gpu_id: int = -1,
    ) -> None:
        self.device = th.device(f"cuda:{gpu_id}") if gpu_id >= 0 else device
        self.num_sims = num_sims

        self.slippage = slippage
        self.delay_step = delay_step
        self.max_holding = 60 * 60 // step_gap
        self.max_position = max_position
        self.step_gap = step_gap
        self.sim_ids = th.arange(self.num_sims, device=self.device)

        args = ConfigData()

        logger.info(
            "TradeSimulator.init -- loading factor array from %s",
            args.predict_ary_path,
        )
        self.factor_ary = np.load(args.predict_ary_path)
        self.factor_ary = th.tensor(self.factor_ary, dtype=th.float32)  # CPU

        logger.info("TradeSimulator.init -- loading price CSV from %s", args.csv_path)
        data_df = pd.read_csv(args.csv_path)

        self.price_ary = data_df[["bids_distance_3", "asks_distance_3", "midpoint"]].values
        self.price_ary[:, 0] = self.price_ary[:, 2] * (1 + self.price_ary[:, 0])
        self.price_ary[:, 1] = self.price_ary[:, 2] * (1 + self.price_ary[:, 1])

        self.llm_signals = data_df[["sentiment_score", "risk_score"]].values

        # Align with the rear of the dataset so price and factors line up.
        self.price_ary = self.price_ary[-self.factor_ary.shape[0] :, :]
        self.llm_signals = self.llm_signals[-self.factor_ary.shape[0] :, :]

        self.price_ary = th.tensor(self.price_ary, dtype=th.float32)  # CPU
        self.llm_signals = th.tensor(self.llm_signals, dtype=th.float32)  # CPU

        self.seq_len = 3600
        self.full_seq_len = self.price_ary.shape[0]
        assert self.price_ary.shape[0] == self.factor_ary.shape[0]
        assert self.llm_signals.shape[0] == self.factor_ary.shape[0]

        # Stateful episode tensors (initialised by reset()).
        self.step_i: int = 0
        self.step_is = th.zeros((num_sims,), dtype=th.long, device=device)
        self.action_int = th.zeros((num_sims,), dtype=th.long, device=device)
        self.rolling_asset = th.zeros((num_sims,), dtype=th.long, device=device)

        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        # Environment metadata consumed by ElegantRL.
        self.env_name = "TradeSimulator-v0"
        self.state_dim = 2 + 8 + 2  # (position, holding) + factor_dim + LLM signals
        self.action_dim = 3  # short, nothing, long
        self.if_discrete = True
        self.max_step = (self.seq_len - num_ignore_step) // step_gap
        self.target_return = +np.inf

        # Stop-loss state.
        self.best_price = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.stop_loss_thresh: float = 1e-3

        logger.info(
            "TradeSimulator ready -- num_sims=%d  full_seq_len=%d  max_step=%d",
            num_sims,
            self.full_seq_len,
            self.max_step,
        )

    def _reset(self, slippage: float | None = None, _if_random: bool = True) -> th.Tensor:
        """Internal reset implementation shared by both environment variants.

        Args:
            slippage: Override the instance slippage value for this episode.
                ``None`` keeps the current value.
            _if_random: Unused flag kept for API compatibility.

        Returns:
            Initial state tensor of shape ``(num_sims, state_dim)``.
        """
        self.slippage = slippage if isinstance(slippage, float) else self.slippage

        num_sims = self.num_sims
        device = self.device

        i0s = np.random.randint(
            self.seq_len,
            self.full_seq_len - self.seq_len * 2,
            size=self.num_sims,
        )
        self.step_i = 0
        self.step_is = th.tensor(i0s, dtype=th.long, device=self.device)
        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        self.best_price = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)

        step_is = self.step_is + self.step_i
        state = self.get_state(step_is_cpu=step_is.to(th.device("cpu")))
        logger.debug("TradeSimulator._reset -- episode started")
        return state

    def _step(
        self, action: th.Tensor, _if_random: bool = True
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, dict]:
        """Internal step implementation.

        Args:
            action: Integer action tensor of shape ``(num_sims, 1)`` with
                values in ``{0, 1, 2}`` mapping to ``{sell, hold, buy}``.
            _if_random: Unused flag kept for API compatibility.

        Returns:
            A tuple of ``(state, reward, terminal, info_dict)`` where
            ``state`` has shape ``(num_sims, state_dim)``, ``reward`` and
            ``terminal`` have shape ``(num_sims,)``.
        """
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i
        step_is_cpu = step_is.to(th.device("cpu"))

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # map (0, 1, 2) -> (-1, 0, +1): sell, hold, buy
        del action

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        mid_price = self.price_ary[step_is_cpu, 2].to(self.device)

        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
        else:
            new_position = (old_position + action_int).clip(-self.max_position, self.max_position)
            action_int = new_position - old_position

            done_mask = (new_position * old_position).lt(0) & old_position.ne(0)
            if done_mask.sum() > 0:
                action_int[done_mask] = -old_position[done_mask]

        # Holding counter.
        self.holding = self.holding + 1
        mask_max_holding = self.holding.gt(self.max_holding)
        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
        self.holding[old_position == 0] = 0

        # Stop-loss logic.
        direction_mask1 = old_position.gt(0)
        if direction_mask1.sum() > 0:
            _best_price = th.max(
                th.stack([self.best_price[direction_mask1], mid_price[direction_mask1]]),
                dim=0,
            )[0]
            self.best_price[direction_mask1] = _best_price

        direction_mask2 = old_position.lt(0)
        if direction_mask2.sum() > 0:
            _best_price = th.min(
                th.stack([self.best_price[direction_mask2], mid_price[direction_mask2]]),
                dim=0,
            )[0]
            self.best_price[direction_mask2] = _best_price

        stop_loss_mask1 = th.logical_and(
            direction_mask1,
            (self.best_price - mid_price).gt(self.stop_loss_thresh),
        )
        stop_loss_mask2 = th.logical_and(
            direction_mask2,
            (mid_price - self.best_price).gt(self.stop_loss_thresh),
        )
        stop_loss_mask = th.logical_or(stop_loss_mask1, stop_loss_mask2)
        if stop_loss_mask.sum() > 0:
            action_int[stop_loss_mask] = -old_position[stop_loss_mask]

        new_position = old_position + action_int

        entry_mask = old_position.eq(0)
        if entry_mask.sum() > 0:
            self.best_price[entry_mask] = mid_price[entry_mask]

        direction = action_int.gt(0)
        cost = action_int * mid_price

        new_cash = old_cash - cost * th.where(direction, 1 + self.slippage, 1 - self.slippage)
        new_asset = new_cash + new_position * mid_price

        reward = new_asset - old_asset

        self.cash = new_cash
        self.asset = new_asset
        self.position = new_position
        self.action_int = action_int

        state = self.get_state(step_is_cpu)
        info_dict: dict = {}
        if truncated:
            terminal = th.ones_like(self.position, dtype=th.bool)
            state = self.reset()
        else:
            terminal = th.zeros_like(self.position, dtype=th.bool)

        return state, reward, terminal, info_dict

    def reset(self, slippage: float | None = None, date_strs: tuple = ()) -> th.Tensor:
        """Reset all parallel environments to random starting positions.

        Args:
            slippage: Optional slippage override.
            date_strs: Unused; present for interface compatibility.

        Returns:
            Initial state tensor of shape ``(num_sims, state_dim)``.
        """
        return self._reset(slippage=slippage, _if_random=True)

    def step(self, action: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, dict]:
        """Advance all parallel environments by one step.

        Args:
            action: Integer action tensor of shape ``(num_sims, 1)``.

        Returns:
            Tuple of ``(state, reward, terminal, info_dict)``.
        """
        return self._step(action, _if_random=True)

    def get_state(self, step_is_cpu: th.Tensor) -> th.Tensor:
        """Construct the current observation tensor for all environments.

        Args:
            step_is_cpu: 1-D long tensor of absolute time indices on CPU.

        Returns:
            State tensor of shape ``(num_sims, state_dim)`` on
            ``self.device``.
        """
        factor_ary = self.factor_ary[step_is_cpu, :].to(self.device)
        llm_signals = self.llm_signals[step_is_cpu, :].to(self.device)

        return th.concat(
            (
                (self.position.float() / self.max_position)[:, None],
                (self.holding.float() / self.max_holding)[:, None],
                factor_ary,
                llm_signals,
            ),
            dim=1,
        )


class EvalTradeSimulator(TradeSimulator):
    """Evaluation variant of :class:`TradeSimulator`.

    Uses a tighter stop-loss threshold and deterministic episode starts
    (sequential rather than random offsets).
    """

    def reset(self, slippage: float | None = None, date_strs: tuple = ()) -> th.Tensor:
        """Reset to a deterministic starting position with a tighter stop-loss.

        Args:
            slippage: Optional slippage override.
            date_strs: Unused; present for interface compatibility.

        Returns:
            Initial state tensor of shape ``(num_sims, state_dim)``.
        """
        self.stop_loss_thresh = 1e-4
        return self._reset(slippage=slippage, _if_random=False)

    def step(self, action: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, dict]:
        """Advance all environments deterministically by one step.

        Args:
            action: Integer action tensor of shape ``(num_sims, 1)``.

        Returns:
            Tuple of ``(state, reward, terminal, info_dict)``.
        """
        return self._step(action, _if_random=False)


def check_simulator() -> None:
    """Run a quick sanity-check of the simulator with random actions.

    Reads the GPU ID from ``sys.argv[1]`` if provided, otherwise defaults
    to CPU.
    """
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    num_sims = 6
    slippage = 0
    step_gap = 2

    sim = TradeSimulator(num_sims=num_sims, step_gap=step_gap, slippage=slippage)
    action_dim = sim.action_dim
    delay_step = sim.delay_step

    reward_ary = th.zeros((num_sims, 4800), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        action = th.randint(action_dim, size=(num_sims, 1), device=device)
        state, reward, done, info_dict = sim.step(action=action)
        reward_ary[:, step_i + delay_step] = reward
        logger.debug("check_simulator.step asset=%s", sim.asset.tolist())

    logger.info("check_simulator.run1 total_reward=%s", reward_ary.sum(dim=1).tolist())
    logger.info("check_simulator.run1 state_shape=%s", list(state.shape))
    assert state.shape == (num_sims, sim.state_dim)

    logger.info("check_simulator.run2 starting")

    reward_ary = th.zeros((num_sims, sim.max_step + delay_step), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        if step_i == 0:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device) - 1
        else:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device)

        state, reward, done, info_dict = sim.step(action=action)
        reward_ary[:, step_i + delay_step] = reward
        if step_i + 2 == sim.max_step:
            logger.debug("check_simulator.penultimate_asset=%s", sim.asset.tolist())

    logger.info("check_simulator.run2 total_reward=%s", reward_ary.sum(dim=1).tolist())
    logger.info("check_simulator.run2 state_shape=%s", list(state.shape))
    assert state.shape == (num_sims, sim.state_dim)
    logger.info("check_simulator.done")


if __name__ == "__main__":
    check_simulator()

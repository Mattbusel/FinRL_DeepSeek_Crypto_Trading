"""Ensemble training orchestration for the LARSA trading system.

Trains multiple DRL agents in sequence and saves each to a dedicated
checkpoint directory.  The resulting ensemble is used at evaluation time
via :mod:`task1_eval`.

Usage::

    python task1_ensemble.py       # CPU
    python task1_ensemble.py 0     # GPU 0
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np
import torch

from config import settings
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config, build_env
from erl_evaluator import Evaluator
from erl_replay_buffer import ReplayBuffer
from exceptions import DataFetchError, ModelError
from logger import get_logger
from trade_simulator import EvalTradeSimulator, TradeSimulator

log = get_logger(__name__)


def can_buy(action: int, mid_price: float, cash: float, current_btc: int) -> tuple[float, int]:
    """Simulate a single buy or sell transaction.

    Args:
        action: ``1`` for buy, ``-1`` for sell, ``0`` for hold.
        mid_price: Current mid-market price of BTC.
        cash: Current cash balance.
        current_btc: Current BTC holdings (integer units).

    Returns:
        A tuple ``(new_cash, new_btc)`` after applying the action.
    """
    if action == 1 and cash > mid_price:
        return cash - mid_price, current_btc + 1
    if action == -1 and current_btc > 0:
        return cash + mid_price, current_btc - 1
    return cash, current_btc


def winloss(action: int, last_price: float, mid_price: float) -> int:
    """Determine whether an action was directionally correct.

    Args:
        action: ``1`` for long, ``-1`` for short, ``0`` for hold.
        last_price: Mid-price at the previous step.
        mid_price: Mid-price at the current step.

    Returns:
        ``1`` for a correct prediction, ``-1`` for incorrect, ``0`` for neutral.
    """
    if action > 0:
        if last_price < mid_price:
            return 1
        if last_price > mid_price:
            return -1
        return 0
    if action < 0:
        if last_price < mid_price:
            return -1
        if last_price > mid_price:
            return 1
        return 0
    return 0


class Ensemble:
    """Trains and saves a collection of DRL agents for ensemble voting.

    Args:
        log_rules: Whether to log detailed action/position distributions
            during training.
        save_path: Root directory under which per-agent checkpoints are saved.
        starting_cash: Initial cash balance used for accounting during
            evaluation episodes embedded in training.
        agent_classes: Ordered list of agent class types to train.
        args: :class:`erl_config.Config` base configuration shared across
            agents.

    Raises:
        DataFetchError: If the evaluation environment cannot be constructed.
    """

    def __init__(
        self,
        log_rules: bool,
        save_path: str,
        starting_cash: float,
        agent_classes: list[type],
        args: Config,
    ) -> None:
        self.log_rules = log_rules
        self.save_path = save_path
        self.starting_cash = starting_cash
        self.current_btc: int = 0
        self.position: list[int] = [0]
        self.btc_assets: list[float] = [0.0]
        self.net_assets: list[float] = [starting_cash]
        self.cash: list[float] = [starting_cash]
        self.agent_classes = agent_classes
        self.from_env_step_is: object | None = None

        self.args = args
        self.agents: list = []
        self.thresh: float = 0.001
        self.num_envs: int = 1
        self.state_dim: int = 2 + 8 + 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eval_env_class = args.eval_env_class
        eval_env_class.num_envs = 1

        eval_env_args: dict = dict(args.eval_env_args)
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = 1

        log.info(
            "ensemble.init",
            save_path=save_path,
            agents=[cls.__name__ for cls in agent_classes],
            device=str(self.device),
        )

        try:
            self.trade_env = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)
        except Exception as exc:
            raise DataFetchError(
                "Failed to build trade environment for ensemble.", cause=exc
            ) from exc

        self.actions: list = []
        self.firstbpi: bool = True

    def save_ensemble(self) -> None:
        """Persist all agent checkpoints under ``save_path/ensemble_models/``.

        Raises:
            ModelError: If any agent cannot be saved.
        """
        ensemble_dir = os.path.join(self.save_path, "ensemble_models")
        os.makedirs(ensemble_dir, exist_ok=True)
        for idx, agent in enumerate(self.agents):
            agent_name = self.agent_classes[idx].__name__
            agent_dir = os.path.join(ensemble_dir, agent_name)
            os.makedirs(agent_dir, exist_ok=True)
            try:
                agent.save_or_load_agent(agent_dir, if_save=True)
            except Exception as exc:
                raise ModelError(
                    f"Failed to save agent '{agent_name}' to '{agent_dir}'.",
                    cause=exc,
                ) from exc
        log.info("ensemble.saved", path=ensemble_dir)

    def ensemble_train(self) -> None:
        """Train every agent class in sequence and save the full ensemble.

        Each agent is trained independently using
        :meth:`train_agent`.  After all agents are trained, the ensemble is
        persisted via :meth:`save_ensemble`.
        """
        log.info("ensemble.train.start", num_agents=len(self.agent_classes))
        args = self.args
        for agent_class in self.agent_classes:
            log.info("agent.train.start", name=agent_class.__name__)
            args.agent_class = agent_class
            agent = self.train_agent(args=args)
            self.agents.append(agent)
            log.info("agent.train.done", name=agent_class.__name__)
        self.save_ensemble()
        log.info("ensemble.train.done")

    def _majority_vote(self, actions: list[int]) -> int:
        """Return the most common action, breaking ties by first occurrence.

        Args:
            actions: List of integer action values from each agent.

        Returns:
            The action integer that received the most votes.
        """
        count: Counter = Counter(actions)
        majority_action, _ = count.most_common(1)[0]
        return majority_action

    def train_agent(self, args: Config) -> object:
        """Train a single agent according to *args* and return it.

        This method builds its own :class:`gym`-style environment, initialises
        a replay buffer, runs the training loop with periodic evaluation, and
        saves the final checkpoint.

        Args:
            args: Configuration object with ``agent_class`` set to the class
                to train.

        Returns:
            The trained agent instance.

        Raises:
            ModelError: If the initial agent state has an unexpected shape.
            DataFetchError: If environment construction fails.
        """
        args.init_before_training()
        torch.set_grad_enabled(False)

        try:
            env = build_env(args.env_class, args.env_args, args.gpu_id)
        except Exception as exc:
            raise DataFetchError("Failed to build training environment.", cause=exc) from exc

        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        state = env.reset()
        if args.num_envs == 1:
            if state.shape != (args.state_dim,):
                raise ModelError(
                    f"Unexpected state shape {state.shape!r}; expected ({args.state_dim},)."
                )
            if not isinstance(state, np.ndarray):
                raise ModelError(f"Expected np.ndarray state; got {type(state).__name__}.")
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            if state.shape != (args.num_envs, args.state_dim):
                raise ModelError(
                    f"state.shape {state.shape!r} != "
                    f"(num_envs={args.num_envs}, state_dim={args.state_dim})."
                )
            if not isinstance(state, torch.Tensor):
                raise ModelError(f"Expected torch.Tensor state; got {type(state).__name__}.")
            state = state.to(agent.device)

        agent.last_state = state.detach()

        if args.if_off_policy:
            buffer: ReplayBuffer | list = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
            buffer_items = agent.explore_env(
                env, args.horizon_len * args.eval_times, if_random=True
            )
            buffer.update(buffer_items)
        else:
            buffer = []

        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)

        cwd = args.cwd
        break_step = args.break_step
        horizon_len = args.horizon_len
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        del args

        if_train = True
        while if_train:
            buffer_items = agent.explore_env(env, horizon_len)

            action_flat = buffer_items[1].flatten()
            action_count = torch.bincount(action_flat).data.cpu().numpy() / action_flat.shape[0]
            action_count = np.ceil(action_count * 998).astype(int)

            position_flat = buffer_items[0][:, :, 0].long().flatten().float()
            position_count = torch.histc(
                position_flat,
                bins=env.max_position * 2 + 1,
                min=-2,
                max=2,
            )
            position_count = position_count.data.cpu().numpy() / position_flat.shape[0]
            position_count = np.ceil(position_count * 998).astype(int)

            log.debug(
                "train.step",
                action_dist=action_count.tolist(),
                position_dist=position_count.tolist(),
            )

            exp_r = float(buffer_items[2].mean().item())
            if if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            evaluator.evaluate_and_save(
                actor=agent.act,
                steps=horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

        elapsed = time.time() - evaluator.start_time
        log.info("train.complete", elapsed_s=round(elapsed, 1), cwd=cwd)

        if hasattr(env, "close"):
            env.close()
        evaluator.save_training_curve_jpg()
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            buffer.save_or_load_history(cwd, if_save=True)

        self.from_env_step_is = env.step_is
        return agent


def run(
    save_path: str,
    agent_list: list[type],
    log_rules: bool = False,
) -> None:
    """Configure and run the full ensemble training pipeline.

    Args:
        save_path: Root directory for saving agent checkpoints.
        agent_list: Agent classes to train.
        log_rules: Whether to emit detailed action/position logs each step.

    Raises:
        DataFetchError: If the environment cannot be constructed.
        ModelError: If a training step produces an invalid model state.
    """
    gpu_id: int = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    num_sims: int = settings.num_sims
    num_ignore_step: int = settings.num_ignore_step
    max_position: int = settings.max_position
    step_gap: int = settings.step_gap
    slippage: float = settings.slippage

    max_step: int = (4800 - num_ignore_step) // step_gap

    env_args: dict = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 2 + 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = tuple(settings.rl_net_dims)

    args.gamma = settings.rl_gamma
    args.explore_rate = settings.rl_explore_rate
    args.state_value_tau = settings.rl_state_value_tau
    args.soft_update_tau = settings.rl_soft_update_tau
    args.learning_rate = settings.rl_learning_rate
    args.batch_size = settings.rl_batch_size
    args.break_step = int(settings.rl_break_step)
    args.buffer_size = int(max_step * 32)
    args.repeat_times = 2
    args.horizon_len = int(max_step * 4)
    args.eval_per_step = int(max_step)
    args.num_workers = 1
    args.save_gap = 8

    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()

    ensemble = Ensemble(
        log_rules,
        save_path,
        settings.starting_cash,
        agent_list,
        args,
    )
    ensemble.ensemble_train()


if __name__ == "__main__":
    run(
        "ensemble_teamname",
        [AgentD3QN, AgentDoubleDQN, AgentDoubleDQN, AgentTwinD3QN],
    )

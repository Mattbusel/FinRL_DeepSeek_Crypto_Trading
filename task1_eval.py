"""Ensemble evaluation runner for the LARSA trading system.

Loads pre-trained agents from disk, runs a single evaluation episode using
the :class:`EnsembleEvaluator`, and prints financial performance metrics.

Usage::

    python task1_eval.py           # CPU
    python task1_eval.py 0         # GPU 0
"""

from __future__ import annotations

import sys
from typing import List, Type

import torch

from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config, build_env
from logger import get_logger
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
from trade_simulator import EvalTradeSimulator
from exceptions import ModelError, DataFetchError

import os
import numpy as np
from collections import Counter

log = get_logger(__name__)


def to_python_number(x: object) -> float:
    """Convert a torch.Tensor or numeric value to a Python float.

    Args:
        x: A :class:`torch.Tensor` scalar or any numeric type.

    Returns:
        A plain Python float.
    """
    if isinstance(x, torch.Tensor):
        return float(x.cpu().item())
    return float(x)  # type: ignore[arg-type]


class EnsembleEvaluator:
    """Loads a trained agent ensemble and runs a single evaluation episode.

    Args:
        save_path: Directory that contains per-agent sub-directories with
            saved weights (as produced by :class:`task1_ensemble.Ensemble`).
        agent_classes: Ordered list of agent class types to load.
        args: :class:`erl_config.Config` instance with environment and
            network settings.

    Raises:
        ModelError: If an agent checkpoint cannot be loaded.
        DataFetchError: If the evaluation environment cannot be constructed.
    """

    def __init__(
        self,
        save_path: str,
        agent_classes: List[Type],
        args: Config,
    ) -> None:
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.args = args
        self.agents: list = []
        self.thresh: float = 0.001
        self.num_envs: int = 1
        self.state_dim: int = 2 + 8 + 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(
            "evaluator.init",
            save_path=save_path,
            agents=[cls.__name__ for cls in agent_classes],
            device=str(self.device),
        )

        try:
            self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
        except Exception as exc:
            raise DataFetchError(
                "Failed to build evaluation environment.", cause=exc
            ) from exc

        self.current_btc: int = 0
        self.cash: list[float] = [args.starting_cash]
        self.btc_assets: list[float] = [0.0]
        self.net_assets: list[float] = [args.starting_cash]
        self.starting_cash: float = args.starting_cash

    def load_agents(self) -> None:
        """Load each agent's weights from its checkpoint directory.

        Raises:
            ModelError: If a checkpoint directory does not exist or weights
                cannot be restored.
        """
        args = self.args
        for agent_class in self.agent_classes:
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            if not os.path.isdir(cwd):
                raise ModelError(
                    f"Agent checkpoint directory not found: '{cwd}'. "
                    "Run task1_ensemble.py first to train and save agents."
                )
            log.info("agent.load", name=agent_name, path=cwd)
            try:
                agent.save_or_load_agent(cwd, if_save=False)
            except Exception as exc:
                raise ModelError(
                    f"Failed to load agent '{agent_name}' from '{cwd}'.", cause=exc
                ) from exc
            self.agents.append(agent)
        log.info("agents.loaded", count=len(self.agents))

    def _ensemble_action(self, actions: list[torch.Tensor]) -> torch.Tensor:
        """Return the majority-vote action across all agents.

        Args:
            actions: List of scalar action tensors from each agent.

        Returns:
            A ``(1, 1)`` int32 :class:`torch.Tensor` with the winning action.
        """
        count: Counter = Counter(int(a.item()) for a in actions)
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)

    def multi_trade(self) -> None:
        """Execute the full evaluation episode and print performance metrics.

        Iterates through all environment steps, collects per-agent actions,
        applies majority voting, simulates cash-and-BTC accounting, and saves
        result arrays to disk beside the agent checkpoints.

        Raises:
            ModelError: If ``load_agents`` has not been called first.
            DataFetchError: If result arrays cannot be saved to disk.
        """
        if not self.agents:
            raise ModelError(
                "No agents loaded. Call load_agents() before multi_trade()."
            )

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()
        last_state = state
        last_price: float = 0.0

        positions: list = []
        action_ints: list[int] = []
        correct_pred: list[int] = []
        current_btcs: list[int] = [self.current_btc]

        log.info("evaluation.start", max_step=trade_env.max_step)

        for step in range(trade_env.max_step):
            actions: list[torch.Tensor] = []
            for agent in agents:
                tensor_state = torch.as_tensor(
                    last_state, dtype=torch.float32, device=agent.device
                )
                q_values = agent.act(tensor_state)
                tensor_action = q_values.argmax(dim=1)
                actions.append(tensor_action.detach().cpu().unsqueeze(1))

            action = self._ensemble_action(actions)
            action_int: int = int(action.item()) - 1

            state, _reward, _done, _ = trade_env.step(action=action)
            action_ints.append(action_int)
            positions.append(trade_env.position)

            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)
            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > float(mid_price):
                new_cash = self.cash[-1] - float(mid_price)
                self.current_btc += 1
            elif action_int < 0 and self.current_btc > 0:
                new_cash = self.cash[-1] + float(mid_price)
                self.current_btc -= 1

            self.cash.append(new_cash)
            self.btc_assets.append(float(self.current_btc * mid_price))
            self.net_assets.append(self.btc_assets[-1] + new_cash)

            last_state = state

            if action_int == 1:
                correct_pred.append(1 if last_price < float(mid_price) else -1 if last_price > float(mid_price) else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < float(mid_price) else 1 if last_price > float(mid_price) else 0)
            else:
                correct_pred.append(0)

            last_price = float(mid_price)
            current_btcs.append(self.current_btc)

        # Persist arrays
        try:
            positions_np = np.array(
                [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in positions]
            )
            np.save(f"{self.save_path}_positions.npy", positions_np)
            np.save(f"{self.save_path}_net_assets.npy", np.array(self.net_assets))
            np.save(f"{self.save_path}_btc_positions.npy", np.array(self.btc_assets))
            np.save(f"{self.save_path}_correct_predictions.npy", np.array(correct_pred))
        except OSError as exc:
            raise DataFetchError(
                f"Failed to save evaluation arrays under '{self.save_path}'.", cause=exc
            ) from exc

        # Compute metrics
        returns = np.diff(self.net_assets) / np.array(self.net_assets[:-1])
        final_sharpe = sharpe_ratio(returns)
        final_mdd = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        log.info(
            "evaluation.metrics",
            sharpe_ratio=round(final_sharpe, 4),
            max_drawdown=round(final_mdd, 4),
            return_over_max_drawdown=round(final_roma, 4),
        )


def run_evaluation(save_path: str, agent_list: List[Type]) -> None:
    """Configure and run the full evaluation pipeline.

    Args:
        save_path: Root directory that contains trained agent checkpoints.
        agent_list: Agent classes to load for ensemble evaluation.

    Raises:
        DataFetchError: If the environment cannot be built.
        ModelError: If agents cannot be loaded or evaluation fails.
    """
    gpu_id: int = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    num_sims: int = 1
    num_ignore_step: int = 60
    max_position: int = 1
    step_gap: int = 2
    slippage: float = 7e-7

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
        "dataset_path": os.path.join(
            os.path.dirname(__file__), "data", "btc_1sec_input.npy"
        ),
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    evaluator = EnsembleEvaluator(save_path, agent_list, args)
    evaluator.load_agents()
    evaluator.multi_trade()


if __name__ == "__main__":
    _save_path = "trained_agents"
    _agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    run_evaluation(_save_path, _agent_list)

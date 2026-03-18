"""ElegantRL training configuration and environment utilities.

This module defines :class:`Config`, the central hyperparameter container for
ElegantRL training runs, along with helper functions :func:`build_env` and
:func:`kwargs_filter` used to construct gym-compatible environments.

Example::

    from erl_config import Config, build_env
    from erl_agent import AgentD3QN
    from trade_simulator import TradeSimulator

    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args={...})
    args.init_before_training()
"""

from __future__ import annotations

import inspect
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch

from logger import get_logger

logger = get_logger(__name__)


class Config:
    """Central configuration container for ElegantRL training runs.

    Stores all hyperparameters for the agent, environment, training loop, and
    evaluation.  Call :meth:`init_before_training` once before starting any
    training loop to seed RNGs and create the checkpoint directory.

    Args:
        agent_class: Class of the RL agent to instantiate.
        env_class: Class of the training environment.
        env_args: Dictionary of keyword arguments for the environment
            constructor.  Must contain ``env_name``, ``state_dim``,
            ``action_dim``, and ``if_discrete`` keys.
    """

    def __init__(
        self,
        agent_class: Optional[Type] = None,
        env_class: Optional[Type] = None,
        env_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.num_envs: Optional[int] = None
        self.agent_class = agent_class
        self.if_off_policy: bool = self.get_if_off_policy()

        self.env_class = env_class
        self.env_args = env_args
        if env_args is None:
            env_args = {
                "env_name": None,
                "num_envs": 1,
                "max_step": 12345,
                "state_dim": None,
                "action_dim": None,
                "if_discrete": None,
            }
        env_args.setdefault("num_envs", 1)
        env_args.setdefault("max_step", 12345)

        self.env_name: Optional[str] = env_args["env_name"]
        self.num_envs = env_args["num_envs"]
        self.max_step: int = env_args["max_step"]
        self.state_dim: Optional[int] = env_args["state_dim"]
        self.action_dim: Optional[int] = env_args["action_dim"]
        self.if_discrete: Optional[bool] = env_args["if_discrete"]

        # Reward shaping
        self.gamma: float = 0.99
        self.reward_scale: float = 2 ** 0

        # Training
        self.net_dims: tuple = (64, 32)
        self.learning_rate: float = 6e-5
        self.clip_grad_norm: float = 3.0
        self.state_value_tau: float = 0.0
        self.soft_update_tau: float = 5e-3
        if self.if_off_policy:
            self.batch_size: int = 64
            self.horizon_len: int = 512
            self.buffer_size: int = int(1e6)
            self.repeat_times: float = 1.0
            self.if_use_per: bool = False
        else:
            self.batch_size = 128
            self.horizon_len = 2048
            self.buffer_size = None  # type: ignore[assignment]
            self.repeat_times = 8.0
            self.if_use_vtrace: bool = False

        # Device
        self.gpu_id: int = 0
        self.num_workers: int = 2
        self.num_threads: int = 8
        self.random_seed: int = 1
        self.learner_gpus: int = 0

        # Evaluation
        self.cwd: Optional[str] = None
        self.if_remove: Optional[bool] = True
        self.break_step: float = np.inf
        self.break_score: float = np.inf
        self.if_keep_save: bool = True
        self.if_over_write: bool = False
        self.if_save_buffer: bool = False

        self.save_gap: int = 8
        self.eval_times: int = 3
        self.eval_per_step: int = int(2e4)
        self.eval_env_class: Optional[Type] = None
        self.eval_env_args: Optional[Dict[str, Any]] = None

    def init_before_training(self) -> None:
        """Seed RNGs, configure PyTorch, and prepare the checkpoint directory.

        Seeds :mod:`numpy` and :mod:`torch` with :attr:`random_seed`, applies
        thread and dtype settings, auto-generates :attr:`cwd` if not set, and
        optionally removes previous run artefacts.
        """
        np.random.seed(self.random_seed % (2 ** 32))
        torch.manual_seed(self.random_seed % (2 ** 32))
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if self.cwd is None:
            self.cwd = (
                f"./{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}"
            )

        if self.if_remove is None:
            self.if_remove = bool(
                input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == "y"
            )
        if self.if_remove:
            shutil.rmtree(self.cwd, ignore_errors=True)
            logger.info("| Arguments Remove cwd: %s", self.cwd)
        else:
            logger.info("| Arguments Keep cwd: %s", self.cwd)
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        """Return ``True`` if the agent class uses an off-policy algorithm.

        Checks the agent class name against a list of known on-policy
        algorithm prefixes.

        Returns:
            ``True`` for off-policy algorithms (DQN, D3QN, SAC, TD3, etc.);
            ``False`` for on-policy algorithms (PPO, A2C, etc.).
        """
        agent_name = self.agent_class.__name__ if self.agent_class else ""
        on_policy_names = ("SARSA", "VPG", "A2C", "A3C", "TRPO", "PPO", "MPO")
        return all(agent_name.find(s) == -1 for s in on_policy_names)

    def print(self) -> None:
        """Pretty-print all configuration attributes to stdout."""
        from pprint import pprint

        pprint(vars(self))


def build_env(
    env_class: Optional[Type] = None,
    env_args: Optional[Dict[str, Any]] = None,
    gpu_id: int = -1,
) -> Any:
    """Instantiate an RL environment, injecting only supported constructor args.

    Sets ``gpu_id`` in *env_args* before construction, then copies standard
    attributes (``env_name``, ``num_envs``, etc.) back onto the returned
    instance so that downstream code can read them uniformly.

    Args:
        env_class: The environment class to instantiate.
        env_args: Keyword argument dictionary for *env_class*.
        gpu_id: GPU device index forwarded to the environment (``-1`` = CPU).

    Returns:
        A constructed environment instance with standard attributes set.
    """
    env_args["gpu_id"] = gpu_id
    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in (
        "env_name",
        "num_envs",
        "max_step",
        "state_dim",
        "action_dim",
        "if_discrete",
    ):
        setattr(env, attr_str, env_args[attr_str])
    return env


def kwargs_filter(
    function: Callable,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Return only the keyword arguments accepted by *function*.

    Inspects the function signature and returns a filtered copy of *kwargs*
    containing only keys that match declared parameter names.

    Args:
        function: Callable whose signature is inspected.
        kwargs: Candidate keyword arguments.

    Returns:
        A sub-dictionary of *kwargs* restricted to *function*'s parameters.
    """
    sign = {val.name for val in inspect.signature(function).parameters.values()}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}


def get_gym_env_args(env: Any, if_print: bool) -> Dict[str, Any]:
    """Extract standard environment metadata from a gym-compatible env.

    Supports both standard OpenAI Gym envs (``0.18.0 <= version <= 0.25.2``)
    and custom environments that expose attributes directly.

    Args:
        env: A gym or gym-compatible environment instance.
        if_print: If ``True``, pretty-print the resulting dict.

    Returns:
        A dictionary with keys ``env_name``, ``num_envs``, ``max_step``,
        ``state_dim``, ``action_dim``, and ``if_discrete``.

    Raises:
        RuntimeError: If the action space is neither Discrete nor Box.
    """
    import gym

    if_gym_standard_env = {
        "unwrapped",
        "observation_space",
        "action_space",
        "spec",
    }.issubset(dir(env))

    if if_gym_standard_env and not hasattr(env, "num_envs"):
        assert "0.18.0" <= gym.__version__ <= "0.25.2"
        env_name: str = env.unwrapped.spec.id
        num_envs: int = getattr(env, "num_envs", 1)
        max_step: int = getattr(env, "_max_episode_steps", 12345)

        state_shape = env.observation_space.shape
        state_dim: Any = state_shape[0] if len(state_shape) == 1 else state_shape

        if_discrete: bool = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:
            action_dim: int = getattr(env.action_space, "n")
        elif isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
            if any(env.action_space.high - 1):
                logger.warning("env.action_space.high %s", env.action_space.high)
            if any(env.action_space.low + 1):
                logger.warning("env.action_space.low %s", env.action_space.low)
        else:
            raise RuntimeError(
                "\n| Error in get_gym_env_info(). Please set these value manually:"
                "\n  `state_dim=int; action_dim=int; if_discrete=bool;`"
                "\n  And keep action_space in range (-1, 1)."
            )
    else:
        env_name = getattr(env, "env_name", "env")
        num_envs = getattr(env, "num_envs", 1)
        max_step = getattr(env, "max_step", 12345)
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args: Dict[str, Any] = {
        "env_name": env_name,
        "num_envs": num_envs,
        "max_step": max_step,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": if_discrete,
    }
    if if_print:
        env_args_str = repr(env_args).replace(",", f",\n{'':11}")
        logger.info("env_args = %s", env_args_str)
    return env_args

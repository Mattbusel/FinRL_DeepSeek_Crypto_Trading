"""DRL agent implementations for the LARSA trading system.

Provides :class:`AgentDoubleDQN`, :class:`AgentD3QN`, and
:class:`AgentTwinD3QN` — all derived from a common Double-DQN base that
supports vectorised environments, soft target-network updates, and optional
state/value normalisation.

Example::

    from erl_agent import AgentD3QN
    from erl_config import Config
    agent = AgentD3QN(net_dims=(128, 128), state_dim=12, action_dim=3, gpu_id=-1)
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from erl_config import Config
from erl_net import QNetTwin, QNetTwinDuel
from erl_replay_buffer import ReplayBuffer
from logger import get_logger

log = get_logger(__name__)


def get_optim_param(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    """Extract all parameter tensors stored in an optimiser's state dict.

    Args:
        optimizer: A :class:`torch.optim.Optimizer` instance.

    Returns:
        A flat list of all :class:`torch.Tensor` objects in the optimiser
        state (e.g. moment estimates for Adam/AdamW).
    """
    params_list: list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


class AgentDoubleDQN:
    """Double Deep Q-Network agent with vectorised environment support.

    Args:
        net_dims: Sequence of hidden-layer widths for the Q-network MLP.
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
        gpu_id: GPU device index (``-1`` for CPU).
        args: :class:`erl_config.Config` with training hyperparameters.
    """

    def __init__(
        self,
        net_dims: list[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ) -> None:
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)
        self.gamma: float = args.gamma
        self.num_envs: int = args.num_envs
        self.batch_size: int = args.batch_size
        self.repeat_times: float = args.repeat_times
        self.reward_scale: float = args.reward_scale
        self.learning_rate: float = args.learning_rate
        self.if_off_policy: bool = args.if_off_policy
        self.clip_grad_norm: float = args.clip_grad_norm
        self.soft_update_tau: float = args.soft_update_tau
        self.state_value_tau: float = args.state_value_tau

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.last_state: Tensor | None = None
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu"
        )

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = (
            cri_class(net_dims, state_dim, action_dim).to(self.device) if cri_class else self.act
        )

        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = (
            torch.optim.AdamW(self.cri.parameters(), self.learning_rate)
            if cri_class
            else self.act_optimizer
        )
        from types import MethodType

        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        self.save_attr_names: set[str] = {
            "act",
            "act_target",
            "act_optimizer",
            "cri",
            "cri_target",
            "cri_optimizer",
        }

        self.act_target = self.cri_target = deepcopy(self.act)
        self.act.explore_rate = getattr(args, "explore_rate", 1 / 32)

    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> tuple[Tensor, Tensor]:
        """Compute critic loss using uniform replay sampling (Double DQN).

        Args:
            buffer: :class:`ReplayBuffer` containing stored transitions.
            batch_size: Number of transitions to sample.

        Returns:
            A tuple ``(obj_critic, q_values)`` where *obj_critic* is the
            combined SmoothL1 loss over both Q-heads and *q_values* are the
            predicted values for the sampled states.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            next_qs = (
                torch.min(*self.cri_target.get_q1_q2(next_ss))
                .max(dim=1, keepdim=True)[0]
                .squeeze(1)
            )
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in self.act.get_q1_q2(states)]
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def save_or_load_agent(self, cwd: str, if_save: bool) -> None:
        """Persist or restore all agent checkpoints from *cwd*.

        Args:
            cwd: Directory path where ``.pth`` files are read from / written to.
            if_save: ``True`` to save; ``False`` to load.
        """
        assert self.save_attr_names.issuperset({"act", "act_target", "act_optimizer"})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(
                    self,
                    attr_name,
                    torch.load(file_path, map_location=self.device, weights_only=True),  # nosec B614
                )

    def explore_env(
        self, env: object, horizon_len: int, if_random: bool = False
    ) -> tuple[Tensor, ...]:
        """Collect trajectories by interacting with a vectorised environment.

        Args:
            env: Vectorised RL environment with ``step()`` and ``reset()``.
            horizon_len: Number of time steps to collect.
            if_random: Use random actions instead of the policy (warm-up).

        Returns:
            A tuple ``(states, actions, rewards, undones)`` where:

            * ``states``  shape ``(horizon_len, num_envs, state_dim)``
            * ``actions`` shape ``(horizon_len, num_envs, 1)``
            * ``rewards`` shape ``(horizon_len, num_envs)``
            * ``undones`` shape ``(horizon_len, num_envs)``
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(
            self.device
        )
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state
        get_action = self.act_target.get_action
        for t in range(horizon_len):
            action = (
                torch.randint(self.action_dim, size=(self.num_envs, 1))
                if if_random
                else get_action(state).detach()
            )
            states[t] = state
            state, reward, done, _ = env.step(action)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state
        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> tuple[float, float]:
        """Perform one round of network updates from replay buffer data.

        Args:
            buffer: Replay buffer containing off-policy transitions.

        Returns:
            A tuple ``(mean_critic_loss, mean_q_value)`` averaged over all
            gradient steps.
        """
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape(
                    (-1,)
                ),
            )

        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> tuple[Tensor, Tensor]:
        """Compute critic loss using uniform sampling (single-head variant).

        Args:
            buffer: :class:`ReplayBuffer` with stored transitions.
            batch_size: Number of transitions to sample.

        Returns:
            A tuple ``(obj_critic, q_values)``.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float) -> None:
        """Perform a soft (Polyak) update of *target_net* from *current_net*.

        Args:
            target_net: Network whose weights are updated towards *current_net*.
            current_net: Source network updated by an optimiser.
            tau: Interpolation coefficient: ``target = (1-tau)*target + tau*current``.
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: torch.optim.Optimizer, objective: Tensor) -> None:
        """Back-propagate *objective* and step *optimizer* with gradient clipping.

        Args:
            optimizer: Optimiser whose parameters are updated.
            objective: Scalar loss tensor to minimise.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(
            parameters=optimizer.param_groups[0]["params"],
            max_norm=self.clip_grad_norm,
        )
        optimizer.step()

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        """Compute discounted cumulative returns via reverse scan.

        Args:
            rewards: Reward tensor of shape ``(horizon_len, num_envs)``.
            undones: Boolean mask (1 = not done) of same shape.

        Returns:
            Discounted return tensor of the same shape as *rewards*.
        """
        returns = torch.empty_like(rewards)
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target(last_state).argmax(dim=1).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor) -> None:
        """Update running mean/std used for state and value normalisation.

        Uses an exponential moving average controlled by
        :attr:`state_value_tau`.  No-ops when *state_value_tau* is zero.

        Args:
            states: Batch of state vectors, shape ``(N, state_dim)``.
            returns: Batch of discounted returns, shape ``(N,)``.
        """
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.act.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentD3QN(AgentDoubleDQN):
    """Dueling Double Deep Q-Network (D3QN) agent.

    Extends :class:`AgentDoubleDQN` with a dueling network architecture
    (:class:`erl_net.QNetTwinDuel`) that separates state-value and advantage
    streams for improved Q-value estimation.

    Args:
        net_dims: Sequence of hidden-layer widths for the dueling Q-network.
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
        gpu_id: GPU device index (``-1`` for CPU).
        args: :class:`erl_config.Config` with training hyperparameters.
    """

    def __init__(
        self,
        net_dims: list[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ) -> None:
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)
        super().__init__(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=gpu_id,
            args=args,
        )


class AgentTwinD3QN(AgentDoubleDQN):
    """Twin-network Double DQN agent using :class:`erl_net.QNetTwin`.

    A variant of D3QN that uses twin Q-heads without the dueling decomposition.

    Args:
        net_dims: Sequence of hidden-layer widths for the twin Q-network.
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
        gpu_id: GPU device index (``-1`` for CPU).
        args: :class:`erl_config.Config` with training hyperparameters.
    """

    def __init__(
        self,
        net_dims: list[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ) -> None:
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)
        super().__init__(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=gpu_id,
            args=args,
        )

"""Tests for AgentDoubleDQN, AgentD3QN, and AgentTwinD3QN."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config
from erl_replay_buffer import ReplayBuffer

STATE_DIM = 8
ACTION_DIM = 3
NET_DIMS = (32, 16)


def _make_config(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> Config:
    env_args = {
        "env_name": "TestEnv-v0",
        "num_envs": 1,
        "max_step": 500,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": True,
    }
    cfg = Config(agent_class=AgentD3QN, env_args=env_args)
    cfg.batch_size = 8
    cfg.buffer_size = 256
    cfg.horizon_len = 16
    return cfg


@pytest.fixture
def ddqn_agent():
    cfg = _make_config()
    return AgentDoubleDQN(
        net_dims=NET_DIMS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gpu_id=-1,
        args=cfg,
    )


@pytest.fixture
def d3qn_agent():
    cfg = _make_config()
    return AgentD3QN(
        net_dims=NET_DIMS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gpu_id=-1,
        args=cfg,
    )


@pytest.fixture
def twin_d3qn_agent():
    cfg = _make_config()
    return AgentTwinD3QN(
        net_dims=NET_DIMS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gpu_id=-1,
        args=cfg,
    )


class TestAgentInitialization:
    def test_ddqn_has_act_network(self, ddqn_agent):
        assert ddqn_agent.act is not None

    def test_ddqn_has_optimizer(self, ddqn_agent):
        assert ddqn_agent.act_optimizer is not None

    def test_ddqn_device_is_cpu(self, ddqn_agent):
        assert ddqn_agent.device.type == "cpu"

    def test_ddqn_state_dim_stored(self, ddqn_agent):
        assert ddqn_agent.state_dim == STATE_DIM

    def test_ddqn_action_dim_stored(self, ddqn_agent):
        assert ddqn_agent.action_dim == ACTION_DIM

    def test_d3qn_initializes(self, d3qn_agent):
        assert d3qn_agent is not None

    def test_twin_d3qn_initializes(self, twin_d3qn_agent):
        assert twin_d3qn_agent is not None

    def test_save_attr_names_contains_required_keys(self, ddqn_agent):
        required = {"act", "act_target", "act_optimizer"}
        assert required.issubset(ddqn_agent.save_attr_names)


class TestActionSelection:
    def _make_state(self, batch: int = 1) -> torch.Tensor:
        return torch.randn(batch, STATE_DIM)

    def test_get_action_shape(self, ddqn_agent):
        state = self._make_state(1)
        action = ddqn_agent.act.get_action(state)
        assert action.shape == (1, 1)

    def test_get_action_within_range(self, ddqn_agent):
        state = self._make_state(4)
        action = ddqn_agent.act.get_action(state)
        assert action.min().item() >= 0
        assert action.max().item() < ACTION_DIM

    def test_get_action_returns_integer_type(self, ddqn_agent):
        state = self._make_state(1)
        action = ddqn_agent.act.get_action(state)
        assert action.dtype in (torch.int32, torch.int64, torch.long)

    def test_d3qn_action_within_range(self, d3qn_agent):
        state = self._make_state(8)
        action = d3qn_agent.act.get_action(state)
        assert action.min().item() >= 0
        assert action.max().item() < ACTION_DIM


class TestSoftUpdate:
    def test_soft_update_moves_target_toward_current(self):
        import torch.nn as nn

        target = nn.Linear(4, 4)
        current = nn.Linear(4, 4)

        with torch.no_grad():
            for p in target.parameters():
                p.fill_(0.0)
            for p in current.parameters():
                p.fill_(1.0)

        AgentDoubleDQN.soft_update(target, current, tau=0.5)

        for p in target.parameters():
            assert abs(p.mean().item() - 0.5) < 1e-6

    def test_soft_update_tau_zero_leaves_target_unchanged(self):
        import torch.nn as nn

        target = nn.Linear(4, 4)
        current = nn.Linear(4, 4)

        with torch.no_grad():
            for p in target.parameters():
                p.fill_(0.0)
            for p in current.parameters():
                p.fill_(1.0)

        AgentDoubleDQN.soft_update(target, current, tau=0.0)

        for p in target.parameters():
            assert abs(p.mean().item()) < 1e-6

    def test_soft_update_tau_one_copies_current(self):
        import torch.nn as nn

        target = nn.Linear(4, 4)
        current = nn.Linear(4, 4)

        with torch.no_grad():
            for p in target.parameters():
                p.fill_(0.0)
            for p in current.parameters():
                p.fill_(2.0)

        AgentDoubleDQN.soft_update(target, current, tau=1.0)

        for p in target.parameters():
            assert abs(p.mean().item() - 2.0) < 1e-6


class TestReplayBufferIntegration:
    """Integration tests: agent + replay buffer together."""

    def _filled_buffer(self) -> ReplayBuffer:
        buf = ReplayBuffer(
            max_size=128,
            state_dim=STATE_DIM,
            action_dim=1,
            gpu_id=-1,
            num_seqs=1,
        )
        states = torch.randn(64, 1, STATE_DIM)
        actions = torch.randint(0, ACTION_DIM, (64, 1, 1)).float()
        rewards = torch.randn(64, 1)
        undones = torch.ones(64, 1)
        buf.update((states, actions, rewards, undones))
        return buf

    def test_get_obj_critic_returns_two_tensors(self, ddqn_agent):
        buf = self._filled_buffer()
        ddqn_agent.last_state = torch.randn(1, STATE_DIM)
        loss, q = ddqn_agent.get_obj_critic(buf, batch_size=8)
        assert loss.ndim == 0  # scalar
        assert q.shape[0] == 8

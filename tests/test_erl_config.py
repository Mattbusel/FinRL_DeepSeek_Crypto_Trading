"""Tests for erl_config.Config and related helpers."""

import sys
import os

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from erl_config import Config


class _FakeOffPolicyAgent:
    """Minimal stand-in for an off-policy agent class."""
    pass


class _FakeOnPolicyAgent:
    """Minimal stand-in for an on-policy agent (name contains 'PPO')."""
    pass

_FakeOnPolicyAgent.__name__ = "AgentPPO"
_FakeOffPolicyAgent.__name__ = "AgentD3QN"


@pytest.fixture()
def off_policy_config():
    """Return a Config wired to a fake off-policy agent."""
    return Config(agent_class=_FakeOffPolicyAgent)


@pytest.fixture()
def on_policy_config():
    """Return a Config wired to a fake on-policy agent."""
    return Config(agent_class=_FakeOnPolicyAgent)


def test_config_has_required_keys(off_policy_config):
    """All mandatory Config attributes must be present."""
    cfg = off_policy_config
    required = [
        "gamma",
        "learning_rate",
        "batch_size",
        "buffer_size",
        "horizon_len",
        "repeat_times",
        "soft_update_tau",
        "clip_grad_norm",
        "net_dims",
        "gpu_id",
        "random_seed",
        "eval_times",
        "eval_per_step",
        "save_gap",
    ]
    for key in required:
        assert hasattr(cfg, key), f"Config missing required attribute: {key}"


def test_learning_rate_positive(off_policy_config):
    """Default learning rate must be strictly positive."""
    assert off_policy_config.learning_rate > 0


def test_batch_size_power_of_two_or_at_least_one(off_policy_config):
    """Batch size must be >= 1 (power-of-two convention)."""
    assert off_policy_config.batch_size >= 1


def test_gamma_in_range(off_policy_config):
    """Discount factor gamma must be in (0, 1]."""
    assert 0 < off_policy_config.gamma <= 1.0


def test_off_policy_flag_for_dqn_agent(off_policy_config):
    """AgentD3QN should be detected as off-policy."""
    assert off_policy_config.if_off_policy is True


def test_on_policy_flag_for_ppo_agent(on_policy_config):
    """AgentPPO should be detected as on-policy."""
    assert on_policy_config.if_off_policy is False


def test_on_policy_has_larger_batch_than_off_policy():
    """On-policy batch size (128) should be >= off-policy default (64)."""
    off = Config(agent_class=_FakeOffPolicyAgent)
    on = Config(agent_class=_FakeOnPolicyAgent)
    assert on.batch_size >= off.batch_size


def test_default_net_dims_is_tuple(off_policy_config):
    """net_dims must be a non-empty sequence."""
    assert len(off_policy_config.net_dims) > 0


def test_soft_update_tau_positive(off_policy_config):
    """Soft-update tau must be strictly positive."""
    assert off_policy_config.soft_update_tau > 0

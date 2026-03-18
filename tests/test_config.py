"""Tests for erl_config.Config and helper utilities."""

from __future__ import annotations

import pytest
import torch

from erl_config import Config, kwargs_filter


class DummyAgentOnPolicy:
    """Stub on-policy agent so get_if_off_policy returns False."""

    __name__ = "PPOAgent"


class DummyAgentOffPolicy:
    """Stub off-policy agent so get_if_off_policy returns True."""

    __name__ = "D3QNAgent"


def _make_env_args(
    state_dim: int = 4,
    action_dim: int = 3,
    if_discrete: bool = True,
) -> dict:
    return {
        "env_name": "TestEnv-v0",
        "num_envs": 1,
        "max_step": 500,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": if_discrete,
    }


class TestConfigDefaults:
    def test_gamma_default(self):
        cfg = Config()
        assert cfg.gamma == 0.99

    def test_off_policy_defaults_batch_size_64(self):
        cfg = Config(agent_class=DummyAgentOffPolicy)
        assert cfg.batch_size == 64

    def test_on_policy_defaults_batch_size_128(self):
        cfg = Config(agent_class=DummyAgentOnPolicy)
        assert cfg.batch_size == 128

    def test_off_policy_flag_true_for_d3qn(self):
        cfg = Config(agent_class=DummyAgentOffPolicy)
        assert cfg.if_off_policy is True

    def test_on_policy_flag_false_for_ppo(self):
        cfg = Config(agent_class=DummyAgentOnPolicy)
        assert cfg.if_off_policy is False

    def test_env_args_sets_state_dim(self):
        cfg = Config(env_args=_make_env_args(state_dim=8))
        assert cfg.state_dim == 8

    def test_env_args_sets_action_dim(self):
        cfg = Config(env_args=_make_env_args(action_dim=5))
        assert cfg.action_dim == 5

    def test_env_name_extracted(self):
        cfg = Config(env_args=_make_env_args())
        assert cfg.env_name == "TestEnv-v0"

    def test_num_envs_defaults_to_one(self):
        args = _make_env_args()
        args.pop("num_envs", None)
        cfg = Config(env_args=args)
        assert cfg.num_envs == 1

    def test_max_step_defaults_when_absent(self):
        args = _make_env_args()
        args.pop("max_step", None)
        cfg = Config(env_args=args)
        assert cfg.max_step == 12345

    def test_net_dims_is_tuple(self):
        cfg = Config()
        assert isinstance(cfg.net_dims, tuple)

    def test_learning_rate_positive(self):
        cfg = Config()
        assert cfg.learning_rate > 0.0

    def test_soft_update_tau_in_range(self):
        cfg = Config()
        assert 0.0 < cfg.soft_update_tau < 1.0


class TestConfigInitBeforeTraining:
    def test_creates_cwd_directory(self, tmp_path):
        cfg = Config(
            agent_class=DummyAgentOffPolicy,
            env_args=_make_env_args(),
        )
        cfg.cwd = str(tmp_path / "run_test")
        cfg.if_remove = False
        cfg.init_before_training()
        assert (tmp_path / "run_test").is_dir()

    def test_removes_existing_cwd_when_if_remove_true(self, tmp_path):
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        sentinel = run_dir / "sentinel.txt"
        sentinel.write_text("data")

        cfg = Config(
            agent_class=DummyAgentOffPolicy,
            env_args=_make_env_args(),
        )
        cfg.cwd = str(run_dir)
        cfg.if_remove = True
        cfg.init_before_training()

        assert run_dir.is_dir()
        assert not sentinel.exists()

    def test_sets_torch_seed(self, tmp_path):
        cfg = Config(
            agent_class=DummyAgentOffPolicy,
            env_args=_make_env_args(),
        )
        cfg.cwd = str(tmp_path / "seed_test")
        cfg.if_remove = False
        cfg.random_seed = 42
        cfg.init_before_training()
        # No assertion beyond "does not raise" -- seeding is a side effect.


class TestKwargsFilter:
    def test_filters_to_matching_params(self):
        def fn(a, b):
            pass

        result = kwargs_filter(fn, {"a": 1, "b": 2, "c": 3})
        assert result == {"a": 1, "b": 2}

    def test_empty_kwargs_returns_empty(self):
        def fn(x):
            pass

        assert kwargs_filter(fn, {}) == {}

    def test_no_matching_params_returns_empty(self):
        def fn(z):
            pass

        assert kwargs_filter(fn, {"a": 1, "b": 2}) == {}

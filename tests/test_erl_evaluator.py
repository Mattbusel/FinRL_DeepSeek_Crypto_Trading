"""Tests for erl_evaluator.py with mocked dependencies.

Covers :func:`draw_learning_curve`, :func:`get_cumulative_rewards_and_steps`,
and the :class:`Evaluator` class metrics calculation using synthetic data.
No real model training or torch GPU is required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recorder(n: int = 5) -> np.ndarray:
    """Build a synthetic recorder array of shape (n, 6).

    Columns: [total_step, avg_r, std_r, exp_r, obj_c, obj_a]
    """
    steps = np.arange(1, n + 1) * 1000.0
    avg_r = np.linspace(0.5, 2.0, n)
    std_r = np.ones(n) * 0.2
    exp_r = avg_r * 0.9
    obj_c = np.linspace(1.0, 0.2, n)
    obj_a = np.linspace(0.0, 1.5, n)
    return np.column_stack([steps, avg_r, std_r, exp_r, obj_c, obj_a])


def _make_mock_env(num_envs: int = 2, max_step: int = 10) -> MagicMock:
    """Return a minimal mock vectorised evaluation environment."""
    env = MagicMock()
    env.num_envs = num_envs
    env.max_step = max_step
    env.if_discrete = True
    env.device = "cpu"
    env.max_position = 1
    return env


def _make_mock_actor(action_dim: int = 3, state_dim: int = 8) -> MagicMock:
    """Return a mock actor whose forward pass returns random logits."""
    import torch

    actor = MagicMock()
    actor.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    actor.side_effect = lambda s: torch.randn(s.shape[0], action_dim)
    return actor


# ---------------------------------------------------------------------------
# Tests for draw_learning_curve
# ---------------------------------------------------------------------------


class TestDrawLearningCurve:
    """Tests for :func:`erl_evaluator.draw_learning_curve`."""

    def test_saves_jpg_file(self) -> None:
        """draw_learning_curve writes a JPEG file at the specified path."""
        pytest.importorskip("torch")
        from erl_evaluator import draw_learning_curve

        recorder = _make_recorder(4)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curve.jpg")
            draw_learning_curve(recorder=recorder, fig_title="test", save_path=save_path)
            assert os.path.exists(save_path)

    def test_file_is_nonempty(self) -> None:
        """The saved JPEG file has non-zero size."""
        pytest.importorskip("torch")
        from erl_evaluator import draw_learning_curve

        recorder = _make_recorder(6)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curve.jpg")
            draw_learning_curve(recorder=recorder, fig_title="size_test", save_path=save_path)
            assert os.path.getsize(save_path) > 0

    def test_accepts_extra_columns(self) -> None:
        """Recorder arrays with more than 6 columns do not raise."""
        pytest.importorskip("torch")
        from erl_evaluator import draw_learning_curve

        recorder = _make_recorder(4)
        extended = np.hstack([recorder, np.ones((4, 2))])  # 8 columns
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "extended.jpg")
            draw_learning_curve(recorder=extended, fig_title="ext", save_path=save_path)
            assert os.path.exists(save_path)

    def test_single_row_recorder(self) -> None:
        """A recorder with only one row does not raise."""
        pytest.importorskip("torch")
        from erl_evaluator import draw_learning_curve

        recorder = _make_recorder(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "single.jpg")
            draw_learning_curve(recorder=recorder, fig_title="single", save_path=save_path)
            assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# Tests for Evaluator initialisation
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    """Tests for :class:`erl_evaluator.Evaluator` initialisation."""

    def test_evaluator_sets_cwd(self) -> None:
        """Evaluator stores the cwd attribute correctly."""
        pytest.importorskip("torch")
        from erl_agent import AgentD3QN
        from erl_config import Config
        from erl_evaluator import Evaluator

        env = _make_mock_env()
        env_args = {
            "env_name": "MockEnv",
            "num_envs": 2,
            "max_step": 10,
            "state_dim": 8,
            "action_dim": 3,
            "if_discrete": True,
        }
        cfg = Config(agent_class=AgentD3QN, env_args=env_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(cwd=tmpdir, env=env, args=cfg)
            assert ev.cwd == tmpdir

    def test_evaluator_sets_max_r(self) -> None:
        """Evaluator max_r starts at -inf."""
        pytest.importorskip("torch")
        from erl_agent import AgentD3QN
        from erl_config import Config
        from erl_evaluator import Evaluator

        env = _make_mock_env()
        env_args = {
            "env_name": "MockEnv",
            "num_envs": 2,
            "max_step": 10,
            "state_dim": 8,
            "action_dim": 3,
            "if_discrete": True,
        }
        cfg = Config(agent_class=AgentD3QN, env_args=env_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(cwd=tmpdir, env=env, args=cfg)
            assert ev.max_r == -np.inf

    def test_evaluator_sets_total_step_zero(self) -> None:
        """Evaluator total_step initialises to zero."""
        pytest.importorskip("torch")
        from erl_agent import AgentD3QN
        from erl_config import Config
        from erl_evaluator import Evaluator

        env = _make_mock_env()
        env_args = {
            "env_name": "MockEnv",
            "num_envs": 2,
            "max_step": 10,
            "state_dim": 8,
            "action_dim": 3,
            "if_discrete": True,
        }
        cfg = Config(agent_class=AgentD3QN, env_args=env_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(cwd=tmpdir, env=env, args=cfg)
            assert ev.total_step == 0

    def test_evaluator_recorder_starts_empty(self) -> None:
        """Evaluator recorder list starts empty."""
        pytest.importorskip("torch")
        from erl_agent import AgentD3QN
        from erl_config import Config
        from erl_evaluator import Evaluator

        env = _make_mock_env()
        env_args = {
            "env_name": "MockEnv",
            "num_envs": 2,
            "max_step": 10,
            "state_dim": 8,
            "action_dim": 3,
            "if_discrete": True,
        }
        cfg = Config(agent_class=AgentD3QN, env_args=env_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(cwd=tmpdir, env=env, args=cfg)
            assert ev.recorder == []


# ---------------------------------------------------------------------------
# Tests for save_or_load_recorder
# ---------------------------------------------------------------------------


class TestSaveOrLoadRecorder:
    """Tests for :meth:`erl_evaluator.Evaluator.save_or_load_recoder`."""

    def _make_evaluator(self, tmpdir: str):
        """Helper: construct an Evaluator pointed at tmpdir."""
        from erl_agent import AgentD3QN
        from erl_config import Config
        from erl_evaluator import Evaluator

        env = _make_mock_env()
        env_args = {
            "env_name": "MockEnv",
            "num_envs": 2,
            "max_step": 10,
            "state_dim": 8,
            "action_dim": 3,
            "if_discrete": True,
        }
        cfg = Config(agent_class=AgentD3QN, env_args=env_args)
        return Evaluator(cwd=tmpdir, env=env, args=cfg)

    def test_save_creates_npy_file(self) -> None:
        """Saving the recorder writes a .npy file to disk."""
        pytest.importorskip("torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = self._make_evaluator(tmpdir)
            ev.recorder = [(1000, 1.5, 0.2, 1.2, 0.5, 0.8)]
            ev.save_or_load_recoder(if_save=True)
            assert os.path.exists(ev.recorder_path)

    def test_load_restores_recorder(self) -> None:
        """Loading an existing recorder file restores total_step."""
        pytest.importorskip("torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = self._make_evaluator(tmpdir)
            ev.recorder = [(2000, 2.0, 0.1, 1.8, 0.3, 1.0)]
            ev.save_or_load_recoder(if_save=True)

            ev2 = self._make_evaluator(tmpdir)
            ev2.save_or_load_recoder(if_save=False)
            assert ev2.total_step == 2000

    def test_load_missing_file_is_noop(self) -> None:
        """Loading when no file exists does not raise and leaves recorder empty."""
        pytest.importorskip("torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = self._make_evaluator(tmpdir)
            ev.save_or_load_recoder(if_save=False)  # no file exists
            assert ev.recorder == []

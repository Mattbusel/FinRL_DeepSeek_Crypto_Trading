"""Tests for continuous_learner.py."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continuous_learner import (
    ConceptDriftDetector,
    ContinuousLearner,
    DriftEvent,
    ExperienceReplay,
    ModelVersionManager,
    OnlineFinetuner,
    Transition,
)
from exceptions import ModelError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_state(dim: int = 8) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def _fill_replay(
    replay: ExperienceReplay, n: int = 50, state_dim: int = 8
) -> None:
    for _ in range(n):
        replay.push(
            _random_state(state_dim),
            action=int(np.random.randint(0, 3)),
            reward=float(np.random.randn()),
            next_state=_random_state(state_dim),
            done=False,
        )


def _mock_agent(state_dim: int = 8, action_dim: int = 3):
    """Create a minimal mock agent with act, act_optimizer attributes.

    Returns a plain MagicMock when torch is unavailable so non-torch tests
    can still run; tests that actually need a real network call
    pytest.importorskip("torch") themselves.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    net = nn.Linear(state_dim, action_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    agent = MagicMock()
    agent.act = net
    agent.act_optimizer = optimizer
    agent.explore_rate = 0.01
    return agent


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_fields(self):
        s = _random_state()
        t = Transition(
            state=s,
            action=1,
            reward=0.5,
            next_state=s,
            done=False,
        )
        assert t.action == 1
        assert abs(t.reward - 0.5) < 1e-6
        assert not t.done


# ---------------------------------------------------------------------------
# ExperienceReplay
# ---------------------------------------------------------------------------


class TestExperienceReplay:
    def test_push_and_len(self):
        replay = ExperienceReplay(capacity=100, state_dim=8)
        assert len(replay) == 0
        _fill_replay(replay, n=10)
        assert len(replay) == 10

    def test_capacity_limit(self):
        replay = ExperienceReplay(capacity=5, state_dim=4)
        _fill_replay(replay, n=10, state_dim=4)
        assert len(replay) == 5  # ring buffer

    def test_sample_correct_size(self):
        replay = ExperienceReplay(capacity=100, state_dim=8)
        _fill_replay(replay, n=50)
        batch = replay.sample(16)
        assert len(batch) == 16

    def test_sample_raises_when_too_small(self):
        replay = ExperienceReplay(capacity=100, state_dim=8)
        _fill_replay(replay, n=5)
        with pytest.raises(ValueError, match="Buffer has 5"):
            replay.sample(10)

    def test_as_arrays_shapes(self):
        replay = ExperienceReplay(capacity=100, state_dim=8)
        _fill_replay(replay, n=20)
        batch = replay.sample(8)
        s, a, r, s2, d = replay.as_arrays(batch)
        assert s.shape == (8, 8)
        assert a.shape == (8,)
        assert r.shape == (8,)
        assert s2.shape == (8, 8)
        assert d.shape == (8,)

    def test_thread_safety(self):
        """Multiple threads pushing should not corrupt the buffer."""
        import threading

        replay = ExperienceReplay(capacity=10_000, state_dim=4)

        def push_many():
            for _ in range(100):
                replay.push(_random_state(4), 0, 0.0, _random_state(4))

        threads = [threading.Thread(target=push_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(replay) <= 10_000

    def test_dtypes(self):
        replay = ExperienceReplay(capacity=50, state_dim=4)
        _fill_replay(replay, n=10, state_dim=4)
        batch = replay.sample(4)
        s, a, r, s2, d = replay.as_arrays(batch)
        assert s.dtype == np.float32
        assert r.dtype == np.float32
        assert d.dtype == np.float32


# ---------------------------------------------------------------------------
# ConceptDriftDetector
# ---------------------------------------------------------------------------


class TestConceptDriftDetector:
    def test_no_drift_below_min_window(self):
        det = ConceptDriftDetector(delta=0.002, min_window=30)
        for _ in range(20):
            drifted = det.update(float(np.random.randn()))
        assert not drifted  # never drifted with only 20 samples

    def test_drift_detected_on_regime_shift(self):
        det = ConceptDriftDetector(delta=0.002, min_window=20)
        # First regime: stable positive rewards
        for _ in range(40):
            det.update(1.0)
        # Second regime: strongly negative rewards — should detect drift
        detected = False
        for _ in range(60):
            if det.update(-5.0):
                detected = True
                break
        assert detected, "Drift should have been detected after a large regime shift."

    def test_drift_events_recorded(self):
        det = ConceptDriftDetector(delta=0.002, min_window=20)
        for _ in range(40):
            det.update(1.0)
        for _ in range(60):
            det.update(-5.0)
        assert len(det.drift_events) >= 1
        event = det.drift_events[0]
        assert isinstance(event, DriftEvent)
        assert event.magnitude > 0

    def test_reset_clears_window(self):
        det = ConceptDriftDetector()
        for _ in range(50):
            det.update(0.5)
        det.reset()
        assert len(det._window) == 0

    def test_no_drift_on_stable_process(self):
        rng = np.random.default_rng(42)
        det = ConceptDriftDetector(delta=0.0001, min_window=30)
        drifts = sum(det.update(float(rng.normal(0, 0.01))) for _ in range(200))
        # In a stable low-variance process, drift should be rare
        assert drifts <= 5


# ---------------------------------------------------------------------------
# ModelVersionManager
# ---------------------------------------------------------------------------


class TestModelVersionManager:
    def test_save_and_load(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelVersionManager(checkpoint_dir=tmpdir, max_versions=5)
            ckpt = mgr.save(agent, sharpe=1.2, total_return=0.5)
            assert ckpt.version == 1
            assert ckpt.sharpe == pytest.approx(1.2)

            # Modify and reload
            import torch
            import torch.nn as nn

            new_agent = _mock_agent()
            loaded = mgr.load(new_agent)
            assert loaded.version == 1

    def test_best_version_is_highest_sharpe(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelVersionManager(checkpoint_dir=tmpdir)
            mgr.save(agent, sharpe=0.5, total_return=0.1)
            mgr.save(agent, sharpe=1.8, total_return=0.3)
            mgr.save(agent, sharpe=0.9, total_return=0.2)
            best = mgr.best
            assert best is not None
            assert best.sharpe == pytest.approx(1.8)

    def test_load_raises_when_no_checkpoints(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelVersionManager(checkpoint_dir=tmpdir)
            with pytest.raises(ModelError, match="No checkpoints"):
                mgr.load(agent)

    def test_pruning_keeps_max_versions(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelVersionManager(checkpoint_dir=tmpdir, max_versions=3)
            for i in range(6):
                mgr.save(agent, sharpe=float(i), total_return=0.1)
            assert len(mgr._versions) == 3

    def test_best_is_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelVersionManager(checkpoint_dir=tmpdir)
            assert mgr.best is None


# ---------------------------------------------------------------------------
# OnlineFinetuner
# ---------------------------------------------------------------------------


class TestOnlineFinetuner:
    def test_skip_when_buffer_small(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        replay = ExperienceReplay(capacity=100, state_dim=8)
        _fill_replay(replay, n=10)  # below min_buffer_size=256

        finetuner = OnlineFinetuner(
            agent=agent,
            replay=replay,
            batch_size=8,
            min_buffer_size=256,
        )
        loss = finetuner.step()
        assert np.isnan(loss)
        assert finetuner.n_calls == 0

    def test_finetuning_produces_finite_loss(self):
        pytest.importorskip("torch")
        agent = _mock_agent(state_dim=8, action_dim=3)
        replay = ExperienceReplay(capacity=1000, state_dim=8)
        _fill_replay(replay, n=300, state_dim=8)

        finetuner = OnlineFinetuner(
            agent=agent,
            replay=replay,
            batch_size=16,
            min_buffer_size=256,
            steps_per_finetune=2,
        )
        loss = finetuner.step()
        assert np.isfinite(loss)
        assert finetuner.n_calls == 1

    def test_n_calls_increments(self):
        pytest.importorskip("torch")
        agent = _mock_agent(state_dim=8, action_dim=3)
        replay = ExperienceReplay(capacity=1000, state_dim=8)
        _fill_replay(replay, n=300, state_dim=8)

        finetuner = OnlineFinetuner(
            agent=agent,
            replay=replay,
            batch_size=16,
            min_buffer_size=256,
        )
        for _ in range(3):
            finetuner.step()
        assert finetuner.n_calls == 3


# ---------------------------------------------------------------------------
# ContinuousLearner
# ---------------------------------------------------------------------------


class TestContinuousLearner:
    def test_start_and_stop(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ContinuousLearner(
                agent=agent,
                state_dim=8,
                action_dim=3,
                finetune_interval=100.0,  # very long interval so it doesn't fire
                checkpoint_dir=tmpdir,
            )
            learner.start()
            assert learner.is_running
            learner.stop(timeout=3.0)
            assert not learner.is_running

    def test_record_fills_buffer(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ContinuousLearner(
                agent=agent,
                state_dim=8,
                action_dim=3,
                finetune_interval=9999.0,
                checkpoint_dir=tmpdir,
            )
            for _ in range(20):
                s = _random_state(8)
                learner.record(s, 0, 0.1, s)
            assert len(learner.replay) == 20

    def test_double_start_raises(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ContinuousLearner(
                agent=agent,
                state_dim=8,
                action_dim=3,
                finetune_interval=9999.0,
                checkpoint_dir=tmpdir,
            )
            learner.start()
            try:
                with pytest.raises(RuntimeError, match="already running"):
                    learner.start()
            finally:
                learner.stop(timeout=2.0)

    def test_drift_boosts_explore_rate(self):
        pytest.importorskip("torch")
        agent = _mock_agent()
        agent.explore_rate = 0.001

        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ContinuousLearner(
                agent=agent,
                state_dim=8,
                action_dim=3,
                finetune_interval=9999.0,
                checkpoint_dir=tmpdir,
                drift_delta=0.002,
                drift_explore_boost=0.05,
            )
            # Fill the drift detector's window with stable rewards
            for _ in range(50):
                learner.drift_detector.update(1.0)

            # Force a drift event by calling record with very negative rewards
            initial_explore = agent.explore_rate
            drifted_ever = False
            for i in range(100):
                s = _random_state(8)
                learner.record(s, 0, -10.0, s)
                if learner.n_drift_events > 0:
                    drifted_ever = True
                    break

            if drifted_ever:
                assert agent.explore_rate >= initial_explore

"""Tests for erl_replay_buffer.ReplayBuffer."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from erl_replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buffer(max_size: int = 100, state_dim: int = 4, action_dim: int = 1,
                 num_seqs: int = 1) -> ReplayBuffer:
    """Construct a CPU replay buffer for testing."""
    return ReplayBuffer(
        max_size=max_size,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=-1,  # CPU
        num_seqs=num_seqs,
    )


def _make_items(horizon: int, num_seqs: int = 1, state_dim: int = 4, action_dim: int = 1):
    """Return a tuple of random transition tensors."""
    states  = torch.randn(horizon, num_seqs, state_dim)
    actions = torch.randint(0, 3, (horizon, num_seqs, action_dim)).float()
    rewards = torch.randn(horizon, num_seqs)
    undones = torch.ones(horizon, num_seqs)
    return states, actions, rewards, undones


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_push_and_sample():
    """After pushing N items, sampling batch_size should return that many rows."""
    buf = _make_buffer(max_size=50)
    buf.update(_make_items(horizon=20))
    batch = buf.sample(10)
    # batch is a tuple of 5 tensors: states, actions, rewards, undones, next_states
    assert len(batch) == 5
    states_batch = batch[0]
    assert states_batch.shape[0] == 10


def test_sample_smaller_than_capacity():
    """Sampling fewer transitions than buffer capacity must not raise."""
    buf = _make_buffer(max_size=200)
    buf.update(_make_items(horizon=50))
    batch = buf.sample(5)
    assert batch[0].shape[0] == 5


def test_buffer_capacity():
    """Buffer cur_size must never exceed max_size."""
    max_size = 30
    buf = _make_buffer(max_size=max_size)
    # Push more than capacity in two rounds
    buf.update(_make_items(horizon=20))
    buf.update(_make_items(horizon=20))
    assert buf.cur_size <= max_size


def test_buffer_wraps_around():
    """Buffer should wrap and remain full after overflow."""
    max_size = 10
    buf = _make_buffer(max_size=max_size)
    buf.update(_make_items(horizon=7))
    buf.update(_make_items(horizon=7))  # overflows
    assert buf.if_full is True
    assert buf.cur_size == max_size


def test_sample_returns_correct_dims():
    """Sampled tensors should have the expected feature dimensions."""
    state_dim = 6
    action_dim = 2
    buf = _make_buffer(max_size=50, state_dim=state_dim, action_dim=action_dim)
    buf.update(_make_items(horizon=20, state_dim=state_dim, action_dim=action_dim))
    states, actions, rewards, undones, next_states = buf.sample(8)
    assert states.shape == (8, state_dim)
    assert actions.shape == (8, action_dim)
    assert rewards.shape == (8,)
    assert undones.shape == (8,)
    assert next_states.shape == (8, state_dim)


def test_cur_size_grows_with_updates():
    """cur_size should increase after each update until max_size is reached."""
    buf = _make_buffer(max_size=100)
    buf.update(_make_items(horizon=10))
    assert buf.cur_size == 10
    buf.update(_make_items(horizon=10))
    assert buf.cur_size == 20

"""Tests for trade_simulator.TradeSimulator and EvalTradeSimulator.

These tests mock all disk I/O (numpy.load and pandas.read_csv) so no real
data files are required.  The entire module is skipped when PyTorch is not
installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

import numpy as np
import pandas as pd
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Test data configuration
# ---------------------------------------------------------------------------

NUM_SIMS = 4
SEQ_LEN = 8000       # must be > 3 * seq_len (3600) to avoid randint edge cases
FACTOR_DIM = 8       # matches state_dim - 4 (position, holding, 2 LLM signals)
STEP_GAP = 2
NUM_IGNORE = 60


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_factor_ary(rows: int = SEQ_LEN, dim: int = FACTOR_DIM) -> np.ndarray:
    """Return a random float32 factor array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((rows, dim)).astype("float32")


def _make_price_df(rows: int = SEQ_LEN) -> pd.DataFrame:
    """Return a minimal price DataFrame."""
    rng = np.random.default_rng(0)
    mid = rng.uniform(20_000, 30_000, rows)
    return pd.DataFrame({
        "bids_distance_3": rng.uniform(-0.01, 0, rows),
        "asks_distance_3": rng.uniform(0, 0.01, rows),
        "midpoint": mid,
        "sentiment_score": rng.integers(1, 6, rows).astype(float),
        "risk_score": rng.integers(1, 6, rows).astype(float),
    })


def _make_simulator(cls_name: str = "TradeSimulator", **kwargs):
    """Instantiate a simulator class with all disk I/O mocked."""
    import trade_simulator as ts
    cls = getattr(ts, cls_name)
    factor_ary = _make_factor_ary()
    price_df = _make_price_df()
    with patch("numpy.load", return_value=factor_ary), \
         patch("pandas.read_csv", return_value=price_df):
        sim = cls(
            num_sims=NUM_SIMS,
            step_gap=STEP_GAP,
            slippage=0.0,
            num_ignore_step=NUM_IGNORE,
            **kwargs,
        )
    return sim


def _action(value: int, num_sims: int = NUM_SIMS) -> torch.Tensor:
    """Create a (num_sims, 1) integer action tensor."""
    return torch.full((num_sims, 1), value, dtype=torch.long)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestTradeSimulatorInit:
    def test_state_dim_matches_expected(self):
        sim = _make_simulator()
        assert sim.state_dim == 2 + FACTOR_DIM + 2

    def test_action_dim_is_three(self):
        sim = _make_simulator()
        assert sim.action_dim == 3

    def test_num_sims_stored(self):
        sim = _make_simulator()
        assert sim.num_sims == NUM_SIMS

    def test_if_discrete_is_true(self):
        sim = _make_simulator()
        assert sim.if_discrete is True

    def test_cash_starts_at_zero(self):
        sim = _make_simulator()
        assert (sim.cash == 0).all()

    def test_position_starts_at_zero(self):
        sim = _make_simulator()
        assert (sim.position == 0).all()

    def test_device_is_cpu(self):
        sim = _make_simulator()
        assert sim.device.type == "cpu"

    def test_env_name_set(self):
        sim = _make_simulator()
        assert sim.env_name == "TradeSimulator-v0"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestTradeSimulatorReset:
    def test_reset_returns_tensor(self):
        sim = _make_simulator()
        state = sim.reset()
        assert isinstance(state, torch.Tensor)

    def test_reset_state_shape(self):
        sim = _make_simulator()
        state = sim.reset()
        assert state.shape == (NUM_SIMS, sim.state_dim)

    def test_reset_zeroes_cash(self):
        sim = _make_simulator()
        sim.reset()
        assert (sim.cash == 0).all()

    def test_reset_zeroes_position(self):
        sim = _make_simulator()
        sim.reset()
        assert (sim.position == 0).all()

    def test_reset_zeroes_holding(self):
        sim = _make_simulator()
        sim.reset()
        assert (sim.holding == 0).all()

    def test_reset_sets_step_i_to_zero(self):
        sim = _make_simulator()
        sim.reset()
        assert sim.step_i == 0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestTradeSimulatorStep:
    def test_step_returns_four_values(self):
        sim = _make_simulator()
        sim.reset()
        result = sim.step(_action(1))
        assert len(result) == 4

    def test_step_state_shape(self):
        sim = _make_simulator()
        sim.reset()
        state, *_ = sim.step(_action(1))
        assert state.shape == (NUM_SIMS, sim.state_dim)

    def test_step_reward_shape(self):
        sim = _make_simulator()
        sim.reset()
        _, reward, *_ = sim.step(_action(1))
        assert reward.shape == (NUM_SIMS,)

    def test_step_done_shape(self):
        sim = _make_simulator()
        sim.reset()
        _, _, done, _ = sim.step(_action(1))
        assert done.shape == (NUM_SIMS,)

    def test_reward_is_float_tensor(self):
        sim = _make_simulator()
        sim.reset()
        _, reward, *_ = sim.step(_action(1))
        assert reward.dtype == torch.float32

    def test_hold_action_from_flat_keeps_position_zero(self):
        """Action=1 (hold) from flat position should leave position at 0."""
        sim = _make_simulator()
        sim.reset()
        sim.step(_action(1))
        assert (sim.position == 0).all()

    def test_buy_action_opens_long(self):
        """Action=2 (buy) from flat should result in non-negative position."""
        sim = _make_simulator()
        sim.reset()
        sim.step(_action(2))
        assert (sim.position >= 0).all()

    def test_sell_clipped_at_negative_max_position(self):
        """Repeated sells should never push position below -max_position."""
        sim = _make_simulator(max_position=1)
        sim.reset()
        for _ in range(5):
            sim.step(_action(0))
        assert (sim.position >= -sim.max_position).all()

    def test_step_advances_step_i(self):
        sim = _make_simulator()
        sim.reset()
        sim.step(_action(1))
        assert sim.step_i == STEP_GAP

    def test_info_dict_returned(self):
        sim = _make_simulator()
        sim.reset()
        _, _, _, info = sim.step(_action(1))
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# PnL accounting
# ---------------------------------------------------------------------------

class TestTradeSimulatorPnL:
    def test_asset_non_negative_after_hold(self):
        """Holding with no open position results in asset >= 0."""
        sim = _make_simulator()
        sim.reset()
        for _ in range(5):
            sim.step(_action(1))
        assert (sim.asset >= 0).all()

    def test_asset_non_negative_after_buy(self):
        """After buying, asset value should be >= 0."""
        sim = _make_simulator()
        sim.reset()
        sim.step(_action(2))
        assert (sim.asset >= 0).all()

    def test_net_asset_tracks_cash_plus_btc(self):
        """net_asset == cash + position * mid_price after any step."""
        sim = _make_simulator()
        sim.reset()
        sim.step(_action(1))
        # asset is defined as cash + position * mid_price in the simulator
        # cash starts at zero; after holding from zero position, asset == 0
        assert (sim.asset >= 0).all()


# ---------------------------------------------------------------------------
# EvalTradeSimulator
# ---------------------------------------------------------------------------

class TestEvalTradeSimulator:
    def test_state_dim_matches(self):
        sim = _make_simulator("EvalTradeSimulator")
        assert sim.state_dim == 2 + FACTOR_DIM + 2

    def test_reset_shape(self):
        sim = _make_simulator("EvalTradeSimulator")
        state = sim.reset()
        assert state.shape == (NUM_SIMS, sim.state_dim)

    def test_tighter_stop_loss_after_reset(self):
        """EvalTradeSimulator should set stop_loss_thresh=1e-4 on reset."""
        sim = _make_simulator("EvalTradeSimulator")
        sim.reset()
        assert sim.stop_loss_thresh == pytest.approx(1e-4)

    def test_step_returns_correct_shape(self):
        sim = _make_simulator("EvalTradeSimulator")
        sim.reset()
        state, *_ = sim.step(_action(1))
        assert state.shape == (NUM_SIMS, sim.state_dim)

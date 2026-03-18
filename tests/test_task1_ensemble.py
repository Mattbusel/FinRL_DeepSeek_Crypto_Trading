"""Tests for task1_ensemble.py.

Coverage targets:
- Ensemble construction and attribute defaults
- _majority_vote logic including tie-breaking
- can_buy transaction logic
- winloss directional correctness
- save_ensemble raises ModelError when an agent cannot save
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from exceptions import ModelError

# ---------------------------------------------------------------------------
# can_buy
# ---------------------------------------------------------------------------


class TestCanBuy:
    def test_buy_deducts_cash_and_increments_btc(self):
        from task1_ensemble import can_buy

        new_cash, new_btc = can_buy(action=1, mid_price=100.0, cash=500.0, current_btc=0)
        assert new_cash == 400.0
        assert new_btc == 1

    def test_sell_adds_cash_and_decrements_btc(self):
        from task1_ensemble import can_buy

        new_cash, new_btc = can_buy(action=-1, mid_price=100.0, cash=500.0, current_btc=2)
        assert new_cash == 600.0
        assert new_btc == 1

    def test_hold_leaves_cash_unchanged(self):
        from task1_ensemble import can_buy

        new_cash, new_btc = can_buy(action=0, mid_price=100.0, cash=500.0, current_btc=1)
        assert new_cash == 500.0
        assert new_btc == 1

    def test_buy_blocked_when_insufficient_cash(self):
        from task1_ensemble import can_buy

        new_cash, new_btc = can_buy(action=1, mid_price=600.0, cash=500.0, current_btc=0)
        assert new_cash == 500.0
        assert new_btc == 0

    def test_sell_blocked_when_no_btc(self):
        from task1_ensemble import can_buy

        new_cash, new_btc = can_buy(action=-1, mid_price=100.0, cash=500.0, current_btc=0)
        assert new_cash == 500.0
        assert new_btc == 0


# ---------------------------------------------------------------------------
# winloss
# ---------------------------------------------------------------------------


class TestWinloss:
    def test_long_correct_when_price_rises(self):
        from task1_ensemble import winloss

        assert winloss(action=1, last_price=100.0, mid_price=105.0) == 1

    def test_long_incorrect_when_price_falls(self):
        from task1_ensemble import winloss

        assert winloss(action=1, last_price=100.0, mid_price=95.0) == -1

    def test_short_correct_when_price_falls(self):
        from task1_ensemble import winloss

        assert winloss(action=-1, last_price=100.0, mid_price=95.0) == 1

    def test_short_incorrect_when_price_rises(self):
        from task1_ensemble import winloss

        assert winloss(action=-1, last_price=100.0, mid_price=105.0) == -1

    def test_hold_always_zero(self):
        from task1_ensemble import winloss

        assert winloss(action=0, last_price=100.0, mid_price=200.0) == 0

    def test_flat_price_returns_zero(self):
        from task1_ensemble import winloss

        assert winloss(action=1, last_price=100.0, mid_price=100.0) == 0


# ---------------------------------------------------------------------------
# Ensemble._majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def _make_ensemble(self):
        """Create a minimal Ensemble instance with all heavy dependencies mocked."""
        mock_args = MagicMock()
        mock_args.eval_env_class = MagicMock()
        mock_args.eval_env_class.num_envs = 1
        mock_args.eval_env_args = {}
        mock_args.gpu_id = -1

        with patch("task1_ensemble.build_env", return_value=MagicMock()):
            from task1_ensemble import Ensemble

            ens = Ensemble(
                log_rules=False,
                save_path="/tmp/test_ensemble",
                starting_cash=1e6,
                agent_classes=[],
                args=mock_args,
            )
        return ens

    def test_clear_majority(self):
        ens = self._make_ensemble()
        assert ens._majority_vote([1, 1, 0]) == 1

    def test_all_same(self):
        ens = self._make_ensemble()
        assert ens._majority_vote([2, 2, 2]) == 2

    def test_two_agents(self):
        ens = self._make_ensemble()
        # With 2 agents result is deterministic via Counter.most_common
        result = ens._majority_vote([0, 1])
        assert result in (0, 1)

    def test_single_agent(self):
        ens = self._make_ensemble()
        assert ens._majority_vote([2]) == 2

    def test_save_ensemble_raises_model_error_on_failure(self):
        """save_ensemble propagates failures as ModelError."""
        mock_args = MagicMock()
        mock_args.eval_env_class = MagicMock()
        mock_args.eval_env_class.num_envs = 1
        mock_args.eval_env_args = {}
        mock_args.gpu_id = -1

        mock_agent_class = MagicMock()
        mock_agent_class.__name__ = "MockAgent"
        bad_agent = MagicMock()
        bad_agent.save_or_load_agent.side_effect = RuntimeError("disk full")

        with patch("task1_ensemble.build_env", return_value=MagicMock()):
            from task1_ensemble import Ensemble

            ens = Ensemble(
                log_rules=False,
                save_path="/tmp/test_save",
                starting_cash=1e6,
                agent_classes=[mock_agent_class],
                args=mock_args,
            )
            ens.agents = [bad_agent]

        with pytest.raises(ModelError):
            ens.save_ensemble()

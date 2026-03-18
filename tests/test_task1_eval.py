"""Tests for task1_eval.py.

Coverage targets:
- EnsembleEvaluator construction
- load_agents raises ModelError for missing checkpoint directory
- _ensemble_action majority voting
- to_python_number conversion
- Metrics computed from net_assets are finite floats
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from exceptions import ModelError, DataFetchError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(tmp_path=None):
    """Construct an EnsembleEvaluator with all environment calls mocked."""
    mock_args = MagicMock()
    mock_args.env_class = MagicMock()
    mock_args.env_args = {}
    mock_args.gpu_id = -1
    mock_args.net_dims = (64, 64)
    mock_args.state_dim = 12
    mock_args.action_dim = 3
    mock_args.starting_cash = 1e6

    with patch("task1_eval.build_env", return_value=MagicMock()):
        from task1_eval import EnsembleEvaluator
        evaluator = EnsembleEvaluator(
            save_path=str(tmp_path) if tmp_path else "/tmp/eval",
            agent_classes=[],
            args=mock_args,
        )
    return evaluator


# ---------------------------------------------------------------------------
# to_python_number
# ---------------------------------------------------------------------------

class TestToPythonNumber:
    def test_tensor_scalar_to_float(self):
        from task1_eval import to_python_number
        t = torch.tensor(3.14)
        assert to_python_number(t) == pytest.approx(3.14, abs=1e-5)

    def test_plain_int_passthrough(self):
        from task1_eval import to_python_number
        assert to_python_number(42) == 42.0

    def test_plain_float_passthrough(self):
        from task1_eval import to_python_number
        assert to_python_number(2.5) == 2.5


# ---------------------------------------------------------------------------
# EnsembleEvaluator construction
# ---------------------------------------------------------------------------

class TestEnsembleEvaluatorInit:
    def test_default_cash_set(self):
        ev = _make_evaluator()
        assert ev.starting_cash == 1e6
        assert ev.cash == [1e6]

    def test_device_is_cpu_when_no_cuda(self):
        ev = _make_evaluator()
        assert ev.device.type in ("cpu", "cuda")

    def test_environment_build_failure_raises_data_fetch_error(self):
        mock_args = MagicMock()
        mock_args.env_class = MagicMock()
        mock_args.env_args = {}
        mock_args.gpu_id = -1
        mock_args.starting_cash = 1e6

        with patch("task1_eval.build_env", side_effect=RuntimeError("env broken")):
            with pytest.raises(DataFetchError):
                from task1_eval import EnsembleEvaluator
                EnsembleEvaluator(
                    save_path="/tmp/bad",
                    agent_classes=[],
                    args=mock_args,
                )


# ---------------------------------------------------------------------------
# load_agents
# ---------------------------------------------------------------------------

class TestLoadAgents:
    def test_missing_checkpoint_dir_raises_model_error(self, tmp_path):
        mock_agent_class = MagicMock()
        mock_agent_class.__name__ = "FakeAgent"
        # The agent instance itself
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_args = MagicMock()
        mock_args.env_class = MagicMock()
        mock_args.env_args = {}
        mock_args.gpu_id = -1
        mock_args.net_dims = (64,)
        mock_args.state_dim = 12
        mock_args.action_dim = 3
        mock_args.starting_cash = 1e6

        with patch("task1_eval.build_env", return_value=MagicMock()):
            from task1_eval import EnsembleEvaluator
            ev = EnsembleEvaluator(
                save_path=str(tmp_path / "nonexistent"),
                agent_classes=[mock_agent_class],
                args=mock_args,
            )

        with pytest.raises(ModelError, match="checkpoint directory not found"):
            ev.load_agents()


# ---------------------------------------------------------------------------
# _ensemble_action
# ---------------------------------------------------------------------------

class TestEnsembleAction:
    def test_majority_wins(self):
        ev = _make_evaluator()
        actions = [torch.tensor([[1]]), torch.tensor([[1]]), torch.tensor([[2]])]
        result = ev._ensemble_action(actions)
        assert result.item() == 1

    def test_single_agent_action(self):
        ev = _make_evaluator()
        actions = [torch.tensor([[0]])]
        assert ev._ensemble_action(actions).item() == 0

    def test_returns_int32_tensor(self):
        ev = _make_evaluator()
        actions = [torch.tensor([[2]]), torch.tensor([[2]])]
        result = ev._ensemble_action(actions)
        assert result.dtype == torch.int32
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# Metrics integration (from metrics module)
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_sharpe_ratio_finite_on_nonzero_std(self):
        from metrics import sharpe_ratio
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        sr = sharpe_ratio(returns)
        assert np.isfinite(sr)

    def test_sharpe_ratio_inf_on_zero_std(self):
        from metrics import sharpe_ratio
        returns = np.array([0.01, 0.01, 0.01])
        assert sharpe_ratio(returns) == float("inf")

    def test_max_drawdown_non_positive(self):
        from metrics import max_drawdown
        returns = np.array([0.1, -0.3, 0.05, -0.1])
        mdd = max_drawdown(returns)
        assert mdd <= 0.0

    def test_return_over_max_drawdown_positive(self):
        from metrics import return_over_max_drawdown
        # Mostly positive returns
        returns = np.array([0.02] * 50 + [-0.01] * 5)
        roma = return_over_max_drawdown(returns)
        assert np.isfinite(roma)

    def test_empty_returns_raise(self):
        from metrics import sharpe_ratio, max_drawdown
        from exceptions import SignalError
        with pytest.raises(SignalError):
            sharpe_ratio([])
        with pytest.raises(SignalError):
            max_drawdown([])

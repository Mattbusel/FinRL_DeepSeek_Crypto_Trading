"""Tests for explainability.py."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from exceptions import ModelError
from explainability import (
    ExplainabilityReport,
    FeatureImportance,
    TradeExplainer,
    _DEFAULT_FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(state_dim: int = 12, action_dim: int = 3) -> MagicMock:
    """Create a mock agent whose act network returns deterministic Q-values."""
    import torch

    agent = MagicMock()
    q_vals = torch.zeros(1, action_dim)
    q_vals[0, 1] = 1.0  # always prefer BUY (action 1)

    # act is an nn.Module-like with __call__ and parameters
    agent.act = MagicMock()
    agent.act.return_value = q_vals
    # Make act callable for use with torch.autograd.grad
    agent.act.__call__ = lambda *a, **kw: q_vals

    return agent


# ---------------------------------------------------------------------------
# FeatureImportance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_abs_shap_positive(self) -> None:
        fi = FeatureImportance(
            feature_name="momentum", shap_value=0.42, direction="bullish", rank=1
        )
        assert fi.abs_shap == pytest.approx(0.42)

    def test_abs_shap_negative(self) -> None:
        fi = FeatureImportance(
            feature_name="volatility", shap_value=-0.33, direction="bearish", rank=2
        )
        assert fi.abs_shap == pytest.approx(0.33)

    def test_direction_bullish(self) -> None:
        fi = FeatureImportance(
            feature_name="sentiment", shap_value=0.1, direction="bullish", rank=1
        )
        assert fi.direction == "bullish"

    def test_direction_bearish(self) -> None:
        fi = FeatureImportance(
            feature_name="spread", shap_value=-0.05, direction="bearish", rank=3
        )
        assert fi.direction == "bearish"


# ---------------------------------------------------------------------------
# ExplainabilityReport
# ---------------------------------------------------------------------------


class TestExplainabilityReport:
    def _make_report(self, action: int = 1) -> ExplainabilityReport:
        features = [
            FeatureImportance("sentiment_score", 0.34, "bullish", 1),
            FeatureImportance("momentum", 0.21, "bullish", 2),
            FeatureImportance("volatility", -0.15, "bearish", 3),
        ]
        return ExplainabilityReport(
            action=action,
            action_label="BUY",
            top_features=features,
            baseline_value=0.1,
            predicted_value=0.9,
            symbol="BTCUSD",
        )

    def test_to_text_contains_action_label(self) -> None:
        report = self._make_report(action=1)
        text = report.to_text()
        assert "BUY" in text

    def test_to_text_contains_symbol(self) -> None:
        report = self._make_report()
        text = report.to_text()
        assert "BTCUSD" in text

    def test_to_text_contains_top_feature(self) -> None:
        report = self._make_report()
        text = report.to_text()
        assert "sentiment_score" in text

    def test_to_text_contains_direction(self) -> None:
        report = self._make_report()
        text = report.to_text()
        assert "bullish" in text or "bearish" in text

    def test_to_text_format_no_symbol(self) -> None:
        features = [FeatureImportance("f1", 0.5, "bullish", 1)]
        report = ExplainabilityReport(
            action=2,
            action_label="SELL",
            top_features=features,
            symbol="",
        )
        text = report.to_text()
        assert "SELL" in text


# ---------------------------------------------------------------------------
# TradeExplainer._to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_from_numpy_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = TradeExplainer._to_numpy(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_from_list(self) -> None:
        result = TradeExplainer._to_numpy([1.0, 2.0])
        assert result.dtype == np.float64
        assert result.shape == (2,)

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
    def test_from_torch_tensor(self) -> None:
        import torch

        t = torch.tensor([[1.0, 2.0, 3.0]])
        result = TradeExplainer._to_numpy(t)
        assert result.shape == (3,)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# TradeExplainer._rank_features
# ---------------------------------------------------------------------------


class TestRankFeatures:
    def test_returns_top_k_features(self) -> None:
        agent = MagicMock()
        explainer = TradeExplainer(agent=agent, top_k=3)
        shap_vals = np.array([0.1, -0.5, 0.3, 0.01, -0.2, 0.4, 0.0, 0.15, -0.05, 0.2, 0.1, 0.05])
        result = explainer._rank_features(shap_vals)
        assert len(result) == 3

    def test_first_feature_has_highest_abs_shap(self) -> None:
        agent = MagicMock()
        explainer = TradeExplainer(agent=agent, top_k=5)
        shap_vals = np.array([0.01, -0.9, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = explainer._rank_features(shap_vals)
        assert result[0].abs_shap >= result[1].abs_shap

    def test_directions_assigned_correctly(self) -> None:
        agent = MagicMock()
        explainer = TradeExplainer(agent=agent, top_k=2)
        shap_vals = np.array([0.5, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = explainer._rank_features(shap_vals)
        directions = {fi.feature_name: fi.direction for fi in result}
        assert directions[_DEFAULT_FEATURE_NAMES[0]] == "bullish"
        assert directions[_DEFAULT_FEATURE_NAMES[1]] == "bearish"

    def test_rank_integers_start_at_one(self) -> None:
        agent = MagicMock()
        explainer = TradeExplainer(agent=agent, top_k=3)
        shap_vals = np.array([0.1, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = explainer._rank_features(shap_vals)
        assert result[0].rank == 1
        assert result[1].rank == 2


# ---------------------------------------------------------------------------
# TradeExplainer.explain_decision — gradient path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestExplainDecisionGradient:
    """Tests that use gradient attribution via a real (tiny) torch model."""

    def _make_linear_agent(self, state_dim: int = 12, action_dim: int = 3) -> MagicMock:
        """Build an agent whose .act is a real tiny nn.Linear for grad tests."""
        import torch
        import torch.nn as nn

        class _TinyNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(state_dim, action_dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        agent = MagicMock()
        agent.act = _TinyNet()
        return agent

    def test_explain_decision_returns_report(self) -> None:
        import torch

        agent = self._make_linear_agent()
        explainer = TradeExplainer(agent=agent, top_k=5)
        state = torch.zeros(1, 12)
        report = explainer.explain_decision(state=state, action=1)
        assert isinstance(report, ExplainabilityReport)
        assert report.action == 1
        assert len(report.top_features) <= 5

    def test_report_action_label_is_buy(self) -> None:
        import torch

        agent = self._make_linear_agent()
        explainer = TradeExplainer(agent=agent, top_k=5)
        state = torch.ones(12)
        report = explainer.explain_decision(state=state, action=1)
        assert report.action_label == "BUY"

    def test_report_action_label_is_sell(self) -> None:
        import torch

        agent = self._make_linear_agent()
        explainer = TradeExplainer(agent=agent, top_k=3)
        state = torch.ones(12)
        report = explainer.explain_decision(state=state, action=2)
        assert report.action_label == "SELL"

    def test_to_text_mentions_because(self) -> None:
        import torch

        agent = self._make_linear_agent()
        explainer = TradeExplainer(agent=agent, top_k=5)
        state = torch.ones(12)
        report = explainer.explain_decision(state=state, action=1)
        text = report.to_text()
        assert "because" in text.lower()

    def test_feature_names_appear_in_report(self) -> None:
        import torch

        names = ["feat_" + str(i) for i in range(12)]
        agent = self._make_linear_agent()
        explainer = TradeExplainer(agent=agent, feature_names=names, top_k=5)
        state = torch.ones(12)
        report = explainer.explain_decision(state=state, action=0)
        reported_names = {fi.feature_name for fi in report.top_features}
        assert reported_names.issubset(set(names))

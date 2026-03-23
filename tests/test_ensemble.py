"""Unit tests for src/ensemble.py — EnsembleVoter and related types."""

from __future__ import annotations

import pytest

from src.ensemble import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    ConfidenceThreshold,
    EnsembleDecision,
    EnsembleVoter,
    ModelPrediction,
    VotingStrategy,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _pred(model_id: str, action: int, confidence: float = 0.7) -> ModelPrediction:
    return ModelPrediction(model_id=model_id, action=action, confidence=confidence, timestamp=0.0)


# ---------------------------------------------------------------------------
# ModelPrediction validation
# ---------------------------------------------------------------------------

class TestModelPrediction:
    def test_valid_sell(self):
        p = _pred("m", ACTION_SELL, 0.9)
        assert p.action == ACTION_SELL
        assert p.confidence == 0.9

    def test_valid_hold(self):
        p = _pred("m", ACTION_HOLD, 0.5)
        assert p.action == ACTION_HOLD

    def test_valid_buy(self):
        p = _pred("m", ACTION_BUY, 0.8)
        assert p.action == ACTION_BUY

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            ModelPrediction(model_id="m", action=2, confidence=0.5, timestamp=0.0)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            ModelPrediction(model_id="m", action=1, confidence=-0.1, timestamp=0.0)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            ModelPrediction(model_id="m", action=1, confidence=1.01, timestamp=0.0)

    def test_confidence_at_boundaries(self):
        _pred("m", ACTION_BUY, 0.0)
        _pred("m", ACTION_SELL, 1.0)


# ---------------------------------------------------------------------------
# EnsembleVoter — registration
# ---------------------------------------------------------------------------

class TestModelRegistration:
    def test_add_model_defaults(self):
        v = EnsembleVoter()
        v.add_model("a")
        assert "a" in v.registered_models

    def test_add_model_custom_weight(self):
        v = EnsembleVoter()
        v.add_model("a", weight=2.5)
        assert "a" in v.registered_models

    def test_invalid_weight_raises(self):
        v = EnsembleVoter()
        with pytest.raises(ValueError):
            v.add_model("a", weight=0.0)
        with pytest.raises(ValueError):
            v.add_model("a", weight=-1.0)

    def test_remove_model(self):
        v = EnsembleVoter()
        v.add_model("a")
        v.remove_model("a")
        assert "a" not in v.registered_models

    def test_remove_missing_raises(self):
        v = EnsembleVoter()
        with pytest.raises(KeyError):
            v.remove_model("ghost")

    def test_empty_predictions_raises(self):
        v = EnsembleVoter()
        with pytest.raises(ValueError):
            v.vote([])


# ---------------------------------------------------------------------------
# MAJORITY strategy
# ---------------------------------------------------------------------------

class TestMajorityVoting:
    def _voter(self):
        v = EnsembleVoter(strategy=VotingStrategy.MAJORITY)
        v.add_model("a")
        v.add_model("b")
        v.add_model("c")
        return v

    def test_clear_majority_buy(self):
        v = self._voter()
        preds = [_pred("a", ACTION_BUY), _pred("b", ACTION_BUY), _pred("c", ACTION_SELL)]
        d = v.vote(preds)
        assert d.action == ACTION_BUY

    def test_clear_majority_sell(self):
        v = self._voter()
        preds = [_pred("a", ACTION_SELL), _pred("b", ACTION_SELL), _pred("c", ACTION_BUY)]
        d = v.vote(preds)
        assert d.action == ACTION_SELL

    def test_unanimous_agreement_rate_one(self):
        v = self._voter()
        preds = [_pred("a", ACTION_BUY), _pred("b", ACTION_BUY), _pred("c", ACTION_BUY)]
        d = v.vote(preds)
        assert d.agreement_rate == pytest.approx(1.0)
        assert d.dissenting_models == []

    def test_dissenting_models_identified(self):
        v = self._voter()
        preds = [_pred("a", ACTION_BUY), _pred("b", ACTION_BUY), _pred("c", ACTION_SELL)]
        d = v.vote(preds)
        assert "c" in d.dissenting_models
        assert d.agreement_rate == pytest.approx(2 / 3)

    def test_tie_break_buy_over_hold(self):
        v = EnsembleVoter(strategy=VotingStrategy.MAJORITY)
        v.add_model("a"); v.add_model("b")
        preds = [_pred("a", ACTION_BUY), _pred("b", ACTION_HOLD)]
        d = v.vote(preds)
        assert d.action == ACTION_BUY

    def test_single_prediction(self):
        v = EnsembleVoter(strategy=VotingStrategy.MAJORITY)
        v.add_model("a")
        d = v.vote([_pred("a", ACTION_SELL)])
        assert d.action == ACTION_SELL
        assert d.agreement_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# WEIGHTED_MAJORITY strategy
# ---------------------------------------------------------------------------

class TestWeightedMajority:
    def test_high_weight_model_overrides_count(self):
        """Two sell votes at weight 1 vs one buy at weight 3."""
        v = EnsembleVoter(strategy=VotingStrategy.WEIGHTED_MAJORITY)
        v.add_model("strong", weight=3.0)
        v.add_model("w1", weight=1.0)
        v.add_model("w2", weight=1.0)
        preds = [_pred("strong", ACTION_BUY), _pred("w1", ACTION_SELL), _pred("w2", ACTION_SELL)]
        d = v.vote(preds)
        assert d.action == ACTION_BUY

    def test_equal_weights_behaves_like_majority(self):
        v = EnsembleVoter(strategy=VotingStrategy.WEIGHTED_MAJORITY)
        v.add_model("a", weight=1.0)
        v.add_model("b", weight=1.0)
        v.add_model("c", weight=1.0)
        preds = [_pred("a", ACTION_HOLD), _pred("b", ACTION_HOLD), _pred("c", ACTION_BUY)]
        d = v.vote(preds)
        assert d.action == ACTION_HOLD

    def test_confidence_is_weighted_fraction(self):
        v = EnsembleVoter(strategy=VotingStrategy.WEIGHTED_MAJORITY)
        v.add_model("a", weight=1.0)
        v.add_model("b", weight=1.0)
        preds = [_pred("a", ACTION_BUY), _pred("b", ACTION_BUY)]
        d = v.vote(preds)
        assert d.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SOFT_VOTING strategy
# ---------------------------------------------------------------------------

class TestSoftVoting:
    def test_high_confidence_wins(self):
        v = EnsembleVoter(strategy=VotingStrategy.SOFT_VOTING)
        v.add_model("a"); v.add_model("b"); v.add_model("c")
        preds = [
            _pred("a", ACTION_SELL, 0.90),  # sell avg = 0.90
            _pred("b", ACTION_BUY,  0.60),  # buy avg = (0.60+0.55)/2 = 0.575
            _pred("c", ACTION_BUY,  0.55),
        ]
        d = v.vote(preds)
        # sell avg > buy avg → sell wins
        assert d.action == ACTION_SELL

    def test_returns_winning_average_as_confidence(self):
        v = EnsembleVoter(strategy=VotingStrategy.SOFT_VOTING)
        v.add_model("a"); v.add_model("b")
        preds = [_pred("a", ACTION_BUY, 0.80), _pred("b", ACTION_BUY, 0.60)]
        d = v.vote(preds)
        assert d.action == ACTION_BUY
        assert d.confidence == pytest.approx(0.70)  # average of 0.80 and 0.60


# ---------------------------------------------------------------------------
# CONFIDENCE_THRESHOLD strategy
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def test_below_threshold_ignored(self):
        v = EnsembleVoter(strategy=ConfidenceThreshold(min_conf=0.75))
        v.add_model("a"); v.add_model("b"); v.add_model("c")
        preds = [
            _pred("a", ACTION_SELL, 0.50),  # filtered out
            _pred("b", ACTION_BUY,  0.80),  # passes
            _pred("c", ACTION_BUY,  0.90),  # passes
        ]
        d = v.vote(preds)
        assert d.action == ACTION_BUY

    def test_all_below_threshold_returns_hold(self):
        v = EnsembleVoter(strategy=ConfidenceThreshold(min_conf=0.99))
        v.add_model("a")
        preds = [_pred("a", ACTION_BUY, 0.5)]
        d = v.vote(preds)
        assert d.action == ACTION_HOLD
        assert d.confidence == pytest.approx(0.0)

    def test_threshold_zero_passes_all(self):
        v = EnsembleVoter(strategy=ConfidenceThreshold(min_conf=0.0))
        v.add_model("a"); v.add_model("b")
        preds = [_pred("a", ACTION_SELL, 0.0), _pred("b", ACTION_SELL, 0.0)]
        d = v.vote(preds)
        assert d.action == ACTION_SELL

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ConfidenceThreshold(min_conf=-0.1)
        with pytest.raises(ValueError):
            ConfidenceThreshold(min_conf=1.1)

    def test_enum_member_default_threshold(self):
        v = EnsembleVoter(strategy=VotingStrategy.CONFIDENCE_THRESHOLD)
        v.add_model("a")
        preds = [_pred("a", ACTION_BUY, 0.6)]
        d = v.vote(preds)
        assert d.action == ACTION_BUY


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_model_uses_last(self):
        v = EnsembleVoter(strategy=VotingStrategy.MAJORITY)
        v.add_model("a")
        preds = [
            _pred("a", ACTION_SELL, 0.9),
            _pred("a", ACTION_BUY,  0.8),  # last → wins
        ]
        d = v.vote(preds)
        # Only one model → single vote for BUY
        assert d.action == ACTION_BUY

    def test_multiple_unique_models(self):
        v = EnsembleVoter(strategy=VotingStrategy.MAJORITY)
        for mid in ["a", "b", "c", "d", "e"]:
            v.add_model(mid)
        preds = [_pred(mid, ACTION_HOLD) for mid in ["a", "b", "c", "d", "e"]]
        d = v.vote(preds)
        assert d.action == ACTION_HOLD
        assert d.agreement_rate == pytest.approx(1.0)

"""Ensemble of RL models with dynamic weighting.

Supports majority vote, weighted average, stacking (best-weight selection),
and best-recent ensemble methods. Weights are updated via exponential decay
after each realized reward.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ModelPrediction
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """Prediction produced by a single sub-model."""

    model_id: str
    action: float       # Continuous action value (e.g. position fraction).
    confidence: float   # Confidence in [0, 1].
    expected_reward: float


# ---------------------------------------------------------------------------
# EnsembleMethod enum
# ---------------------------------------------------------------------------

class EnsembleMethod(Enum):
    MAJORITY_VOTE    = auto()
    WEIGHTED_AVERAGE = auto()
    STACKING         = auto()
    BEST_RECENT      = auto()


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """Ensemble of RL models with adaptive weight updates."""

    # Exponential moving average decay for weight updates.
    _EMA_ALPHA = 0.1

    def __init__(self) -> None:
        self._weights: Dict[str, float] = {}
        self._realized_rewards: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def add_model(self, model_id: str, initial_weight: float = 1.0) -> None:
        """Register a new sub-model with an initial weight."""
        self._weights[model_id] = max(initial_weight, 0.0)
        self._realized_rewards[model_id] = []

    # ------------------------------------------------------------------
    # Prediction aggregation
    # ------------------------------------------------------------------

    def predict(
        self,
        predictions: List[ModelPrediction],
        method: EnsembleMethod,
    ) -> ModelPrediction:
        """Aggregate sub-model predictions into a single ensemble prediction.

        Raises ValueError if predictions is empty.
        """
        if not predictions:
            raise ValueError("predictions list is empty")

        if method == EnsembleMethod.MAJORITY_VOTE:
            return self._majority_vote(predictions)
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(predictions)
        elif method == EnsembleMethod.STACKING:
            return self._stacking(predictions)
        else:  # BEST_RECENT
            return self._best_recent(predictions)

    def _majority_vote(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Sign-based majority vote: aggregate sign of actions."""
        positive = sum(1 for p in predictions if p.action >= 0)
        negative = len(predictions) - positive
        consensus_sign = 1.0 if positive >= negative else -1.0

        avg_magnitude = sum(abs(p.action) for p in predictions) / len(predictions)
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        avg_reward = sum(p.expected_reward for p in predictions) / len(predictions)

        return ModelPrediction(
            model_id="ensemble_majority_vote",
            action=consensus_sign * avg_magnitude,
            confidence=avg_confidence,
            expected_reward=avg_reward,
        )

    def _weighted_average(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Confidence-weighted average of actions."""
        total_weight = sum(
            self._weights.get(p.model_id, 1.0) * p.confidence
            for p in predictions
        )
        if total_weight == 0.0:
            total_weight = 1.0

        action = sum(
            self._weights.get(p.model_id, 1.0) * p.confidence * p.action
            for p in predictions
        ) / total_weight

        confidence = sum(
            self._weights.get(p.model_id, 1.0) * p.confidence
            for p in predictions
        ) / total_weight

        expected_reward = sum(
            self._weights.get(p.model_id, 1.0) * p.expected_reward
            for p in predictions
        ) / total_weight

        return ModelPrediction(
            model_id="ensemble_weighted_avg",
            action=action,
            confidence=min(confidence, 1.0),
            expected_reward=expected_reward,
        )

    def _stacking(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Select the prediction from the highest-weight model."""
        best = max(
            predictions,
            key=lambda p: self._weights.get(p.model_id, 1.0),
        )
        return ModelPrediction(
            model_id="ensemble_stacking",
            action=best.action,
            confidence=best.confidence,
            expected_reward=best.expected_reward,
        )

    def _best_recent(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Select the prediction from the model with best recent average reward."""
        def recent_avg(p: ModelPrediction) -> float:
            rewards = self._realized_rewards.get(p.model_id, [])
            if not rewards:
                return self._weights.get(p.model_id, 0.0)
            window = rewards[-10:]
            return sum(window) / len(window)

        best = max(predictions, key=recent_avg)
        return ModelPrediction(
            model_id="ensemble_best_recent",
            action=best.action,
            confidence=best.confidence,
            expected_reward=best.expected_reward,
        )

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, model_id: str, realized_reward: float) -> None:
        """Update weight for model_id using exponential decay.

        w_new = (1 - alpha) * w_old + alpha * realized_reward
        Negative rewards are allowed (will pull weight down).
        """
        if model_id not in self._weights:
            self._weights[model_id] = 1.0
        self._realized_rewards.setdefault(model_id, []).append(realized_reward)
        old_w = self._weights[model_id]
        self._weights[model_id] = (1.0 - self._EMA_ALPHA) * old_w + self._EMA_ALPHA * realized_reward

    def normalize_weights(self) -> None:
        """Normalize all weights so they sum to 1.0."""
        total = sum(max(w, 0.0) for w in self._weights.values())
        if total == 0.0:
            n = len(self._weights)
            for mid in self._weights:
                self._weights[mid] = 1.0 / n if n > 0 else 0.0
        else:
            for mid in self._weights:
                self._weights[mid] = max(self._weights[mid], 0.0) / total

    def best_model(self) -> str:
        """Return the model_id with the highest weight."""
        if not self._weights:
            raise ValueError("No models registered")
        return max(self._weights, key=lambda mid: self._weights[mid])

    def weight_entropy(self) -> float:
        """Shannon entropy of the normalized weight distribution.

        High entropy = diverse ensemble, low entropy = dominated by one model.
        """
        if not self._weights:
            return 0.0
        total = sum(max(w, 0.0) for w in self._weights.values())
        if total == 0.0:
            return 0.0
        probs = [max(w, 0.0) / total for w in self._weights.values()]
        return -sum(p * math.log(p + 1e-12) for p in probs)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEnsembleModel(unittest.TestCase):
    """Unit tests for EnsembleModel."""

    def _make_preds(self) -> List[ModelPrediction]:
        return [
            ModelPrediction("model_a", action=0.8,  confidence=0.9, expected_reward=0.5),
            ModelPrediction("model_b", action=-0.3, confidence=0.6, expected_reward=0.2),
            ModelPrediction("model_c", action=0.5,  confidence=0.7, expected_reward=0.4),
        ]

    def setUp(self) -> None:
        self.ensemble = EnsembleModel()
        self.ensemble.add_model("model_a", initial_weight=1.0)
        self.ensemble.add_model("model_b", initial_weight=1.0)
        self.ensemble.add_model("model_c", initial_weight=1.0)

    def test_add_model_stores_weight(self) -> None:
        e = EnsembleModel()
        e.add_model("x", 2.5)
        self.assertAlmostEqual(e.weights["x"], 2.5)

    def test_predict_majority_vote_positive_majority(self) -> None:
        preds = self._make_preds()
        result = self.ensemble.predict(preds, EnsembleMethod.MAJORITY_VOTE)
        # 2 positive, 1 negative → positive action
        self.assertGreater(result.action, 0.0)

    def test_predict_majority_vote_model_id(self) -> None:
        preds = self._make_preds()
        result = self.ensemble.predict(preds, EnsembleMethod.MAJORITY_VOTE)
        self.assertEqual(result.model_id, "ensemble_majority_vote")

    def test_predict_weighted_average_model_id(self) -> None:
        result = self.ensemble.predict(self._make_preds(), EnsembleMethod.WEIGHTED_AVERAGE)
        self.assertEqual(result.model_id, "ensemble_weighted_avg")

    def test_predict_weighted_average_action_bounded(self) -> None:
        result = self.ensemble.predict(self._make_preds(), EnsembleMethod.WEIGHTED_AVERAGE)
        self.assertGreater(result.action, -1.0)
        self.assertLess(result.action, 1.0)

    def test_predict_stacking_picks_highest_weight(self) -> None:
        self.ensemble._weights["model_b"] = 5.0
        result = self.ensemble.predict(self._make_preds(), EnsembleMethod.STACKING)
        self.assertAlmostEqual(result.action, -0.3)

    def test_predict_best_recent_no_history_uses_weight(self) -> None:
        # No history → falls back to weight ordering.
        self.ensemble._weights["model_c"] = 10.0
        result = self.ensemble.predict(self._make_preds(), EnsembleMethod.BEST_RECENT)
        self.assertAlmostEqual(result.action, 0.5)

    def test_predict_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            self.ensemble.predict([], EnsembleMethod.MAJORITY_VOTE)

    def test_update_weights_ema(self) -> None:
        old = self.ensemble.weights["model_a"]
        self.ensemble.update_weights("model_a", 2.0)
        expected = (1.0 - 0.1) * old + 0.1 * 2.0
        self.assertAlmostEqual(self.ensemble.weights["model_a"], expected)

    def test_update_weights_stores_reward(self) -> None:
        self.ensemble.update_weights("model_b", 0.7)
        self.assertIn(0.7, self.ensemble._realized_rewards["model_b"])

    def test_normalize_weights_sum_to_one(self) -> None:
        self.ensemble._weights = {"model_a": 3.0, "model_b": 1.0, "model_c": 1.0}
        self.ensemble.normalize_weights()
        total = sum(self.ensemble.weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_normalize_weights_zero_total(self) -> None:
        self.ensemble._weights = {"a": 0.0, "b": 0.0}
        self.ensemble.normalize_weights()
        total = sum(self.ensemble.weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_best_model_returns_highest_weight(self) -> None:
        self.ensemble._weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}
        self.assertEqual(self.ensemble.best_model(), "model_a")

    def test_best_model_raises_when_no_models(self) -> None:
        e = EnsembleModel()
        with self.assertRaises(ValueError):
            e.best_model()

    def test_weight_entropy_uniform_is_high(self) -> None:
        # Uniform weights → maximum entropy.
        self.ensemble._weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        entropy = self.ensemble.weight_entropy()
        self.assertGreater(entropy, math.log(2))

    def test_weight_entropy_concentrated_is_low(self) -> None:
        self.ensemble._weights = {"a": 1000.0, "b": 0.001, "c": 0.001}
        entropy = self.ensemble.weight_entropy()
        self.assertLess(entropy, 0.1)

    def test_weight_entropy_no_models(self) -> None:
        e = EnsembleModel()
        self.assertAlmostEqual(e.weight_entropy(), 0.0)

    def test_update_weights_new_model_auto_registered(self) -> None:
        e = EnsembleModel()
        e.update_weights("new_model", 1.5)
        self.assertIn("new_model", e.weights)


if __name__ == "__main__":
    unittest.main()

"""Ensemble Voting System for multi-model trading signal aggregation.

Combines predictions from multiple trading models using configurable
voting strategies to produce a single consensus trading decision.

Typical usage::

    from src.ensemble import (
        EnsembleVoter,
        ModelPrediction,
        VotingStrategy,
        ConfidenceThreshold,
    )

    voter = EnsembleVoter(strategy=VotingStrategy.WEIGHTED_MAJORITY)
    voter.add_model("model_a", weight=1.5)
    voter.add_model("model_b", weight=1.0)
    voter.add_model("model_c", weight=0.8)

    predictions = [
        ModelPrediction(model_id="model_a", action=1,  confidence=0.82, timestamp=1700000000.0),
        ModelPrediction(model_id="model_b", action=1,  confidence=0.65, timestamp=1700000000.0),
        ModelPrediction(model_id="model_c", action=-1, confidence=0.55, timestamp=1700000000.0),
    ]

    decision = voter.vote(predictions)
    print(decision.action)          # 1  (buy)
    print(decision.agreement_rate)  # 0.667
    print(decision.dissenting_models)  # ["model_c"]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
ACTION_SELL = -1
ACTION_HOLD = 0
ACTION_BUY = 1
_VALID_ACTIONS = {ACTION_SELL, ACTION_HOLD, ACTION_BUY}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """A single prediction from one model.

    Attributes:
        model_id:   Unique identifier for the originating model.
        action:     Trading action: -1 = sell, 0 = hold, 1 = buy.
        confidence: Prediction confidence in [0.0, 1.0].
        timestamp:  Unix epoch seconds when the prediction was generated.
    """
    model_id: str
    action: int           # -1, 0, or 1
    confidence: float     # [0.0, 1.0]
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {_VALID_ACTIONS}, got {self.action!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")


@dataclass
class EnsembleDecision:
    """The aggregated decision produced by :class:`EnsembleVoter`.

    Attributes:
        action:            Consensus action (-1, 0, or 1).
        confidence:        Aggregate confidence score (strategy-dependent).
        agreement_rate:    Fraction of models (by count) that agree with
                           *action*.  1.0 = unanimous.
        dissenting_models: List of model IDs that voted differently.
    """
    action: int
    confidence: float
    agreement_rate: float
    dissenting_models: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

class VotingStrategy(Enum):
    """Available voting strategies for :class:`EnsembleVoter`."""

    MAJORITY = "majority"
    """Simple majority: the most frequent action wins.
    Ties are broken deterministically: buy > hold > sell."""

    WEIGHTED_MAJORITY = "weighted_majority"
    """Weighted majority: votes are weighted by the per-model weight registered
    via :meth:`EnsembleVoter.add_model`.  Ties broken the same way."""

    SOFT_VOTING = "soft_voting"
    """Soft voting: average confidence per action class; the class with the
    highest average wins."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    """Only include predictions whose confidence exceeds the configured
    minimum.  If no predictions survive the filter, HOLD is returned."""


@dataclass
class ConfidenceThreshold:
    """Wrapper to parameterise the :attr:`VotingStrategy.CONFIDENCE_THRESHOLD` strategy.

    Pass an instance of this to :class:`EnsembleVoter` instead of the enum
    member when you need a custom threshold.

    Example::

        voter = EnsembleVoter(strategy=ConfidenceThreshold(min_conf=0.70))
    """
    min_conf: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_conf <= 1.0):
            raise ValueError(f"min_conf must be in [0, 1], got {self.min_conf!r}")


# ---------------------------------------------------------------------------
# EnsembleVoter
# ---------------------------------------------------------------------------

class EnsembleVoter:
    """Aggregates predictions from multiple models into a single decision.

    Parameters:
        strategy: One of :class:`VotingStrategy` enum members, or an instance
                  of :class:`ConfidenceThreshold` for a parameterised threshold.

    Example::

        voter = EnsembleVoter(strategy=VotingStrategy.SOFT_VOTING)
        voter.add_model("dqn_1")
        voter.add_model("d3qn", weight=2.0)
        decision = voter.vote(predictions)
    """

    def __init__(
        self,
        strategy: VotingStrategy | ConfidenceThreshold = VotingStrategy.MAJORITY,
    ) -> None:
        self._strategy = strategy
        self._weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_model(self, model_id: str, weight: float = 1.0) -> None:
        """Register a model with an optional voting weight.

        Args:
            model_id: Unique identifier for the model.
            weight:   Relative weight (must be > 0).  Only used by
                      :attr:`VotingStrategy.WEIGHTED_MAJORITY`.
        """
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight!r}")
        self._weights[model_id] = weight

    def remove_model(self, model_id: str) -> None:
        """Deregister a model.  Raises :exc:`KeyError` if not found."""
        del self._weights[model_id]

    @property
    def registered_models(self) -> List[str]:
        """Return list of registered model IDs."""
        return list(self._weights.keys())

    def vote(self, predictions: List[ModelPrediction]) -> EnsembleDecision:
        """Aggregate *predictions* according to the configured strategy.

        Args:
            predictions: One or more :class:`ModelPrediction` objects.  Each
                         model may appear at most once; if duplicates are
                         present, the last prediction for that model_id wins.

        Returns:
            An :class:`EnsembleDecision` with the consensus action.

        Raises:
            ValueError: If *predictions* is empty.
        """
        if not predictions:
            raise ValueError("predictions list must not be empty")

        # Deduplicate: last prediction per model wins.
        seen: Dict[str, ModelPrediction] = {}
        for p in predictions:
            seen[p.model_id] = p
        unique_preds = list(seen.values())

        strategy = self._strategy

        if isinstance(strategy, ConfidenceThreshold):
            return self._confidence_threshold_vote(unique_preds, strategy.min_conf)
        elif strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(unique_preds, weighted=False)
        elif strategy == VotingStrategy.WEIGHTED_MAJORITY:
            return self._majority_vote(unique_preds, weighted=True)
        elif strategy == VotingStrategy.SOFT_VOTING:
            return self._soft_vote(unique_preds)
        elif strategy == VotingStrategy.CONFIDENCE_THRESHOLD:
            # Default threshold when used as plain enum member.
            return self._confidence_threshold_vote(unique_preds, min_conf=0.5)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _majority_vote(
        self, predictions: List[ModelPrediction], weighted: bool
    ) -> EnsembleDecision:
        """Simple or weighted majority vote."""
        scores: Dict[int, float] = {ACTION_SELL: 0.0, ACTION_HOLD: 0.0, ACTION_BUY: 0.0}

        for p in predictions:
            w = self._weights.get(p.model_id, 1.0) if weighted else 1.0
            scores[p.action] += w

        winning_action = self._break_tie(scores)
        total = sum(scores.values())
        confidence = scores[winning_action] / total if total > 0 else 0.0

        agreement_rate, dissenters = self._compute_agreement(predictions, winning_action)
        return EnsembleDecision(
            action=winning_action,
            confidence=confidence,
            agreement_rate=agreement_rate,
            dissenting_models=dissenters,
        )

    def _soft_vote(self, predictions: List[ModelPrediction]) -> EnsembleDecision:
        """Average confidence per action class; highest average wins."""
        sums: Dict[int, float] = {ACTION_SELL: 0.0, ACTION_HOLD: 0.0, ACTION_BUY: 0.0}
        counts: Dict[int, int] = {ACTION_SELL: 0, ACTION_HOLD: 0, ACTION_BUY: 0}

        for p in predictions:
            sums[p.action] += p.confidence
            counts[p.action] += 1

        averages: Dict[int, float] = {
            a: (sums[a] / counts[a]) if counts[a] > 0 else 0.0
            for a in _VALID_ACTIONS
        }

        winning_action = self._break_tie(averages)
        confidence = averages[winning_action]
        agreement_rate, dissenters = self._compute_agreement(predictions, winning_action)

        return EnsembleDecision(
            action=winning_action,
            confidence=confidence,
            agreement_rate=agreement_rate,
            dissenting_models=dissenters,
        )

    def _confidence_threshold_vote(
        self, predictions: List[ModelPrediction], min_conf: float
    ) -> EnsembleDecision:
        """Majority vote over predictions above the confidence threshold."""
        filtered = [p for p in predictions if p.confidence >= min_conf]

        if not filtered:
            # No predictions survive the filter — return HOLD with 0 confidence.
            return EnsembleDecision(
                action=ACTION_HOLD,
                confidence=0.0,
                agreement_rate=0.0,
                dissenting_models=[p.model_id for p in predictions],
            )

        # Simple majority among survivors.
        result = self._majority_vote(filtered, weighted=False)
        # Agreement rate is still computed over *all* input predictions.
        agreement_rate, dissenters = self._compute_agreement(predictions, result.action)
        return EnsembleDecision(
            action=result.action,
            confidence=result.confidence,
            agreement_rate=agreement_rate,
            dissenting_models=dissenters,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _break_tie(scores: Dict[int, float]) -> int:
        """Return the action with the highest score, breaking ties buy > hold > sell."""
        tie_priority = [ACTION_BUY, ACTION_HOLD, ACTION_SELL]
        best_score = max(scores.values())
        for action in tie_priority:
            if scores[action] == best_score:
                return action
        return ACTION_HOLD  # unreachable

    @staticmethod
    def _compute_agreement(
        predictions: List[ModelPrediction], winning_action: int
    ) -> tuple[float, List[str]]:
        """Return (agreement_rate, dissenting_model_ids)."""
        n = len(predictions)
        if n == 0:
            return 0.0, []
        dissenters = [p.model_id for p in predictions if p.action != winning_action]
        agreement_rate = (n - len(dissenters)) / n
        return agreement_rate, dissenters

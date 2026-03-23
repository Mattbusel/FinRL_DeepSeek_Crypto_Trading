"""Multi-agent RL coordination framework for crypto trading.

Implements:
- ``AgentRole``           — enum of distinct trading strategies.
- ``AgentSignal``         — a timestamped trade signal from one agent.
- ``MultiAgentCoordinator`` — aggregates signals from registered agents.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# AgentRole
# ---------------------------------------------------------------------------


class AgentRole(Enum):
    """Distinct trading strategy roles for agent specialization."""

    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERTER = "mean_reverter"
    MOMENTUM = "momentum"
    VOLATILITY_TRADER = "volatility_trader"


# ---------------------------------------------------------------------------
# AgentSignal
# ---------------------------------------------------------------------------


@dataclass
class AgentSignal:
    """A trade signal submitted by one agent.

    Attributes:
        agent_id:   Unique string identifier of the submitting agent.
        role:       The agent's trading-strategy role.
        action:     +1 = buy, −1 = sell, 0 = hold.
        confidence: Score in [0.0, 1.0] indicating signal conviction.
        timestamp:  POSIX timestamp (seconds) when the signal was generated.
    """

    agent_id: str
    role: AgentRole
    action: int  # +1, -1, 0
    confidence: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MultiAgentCoordinator
# ---------------------------------------------------------------------------


@dataclass
class _AgentMeta:
    role: AgentRole
    weight: float
    correct_calls: int = 0
    total_calls: int = 0


class MultiAgentCoordinator:
    """Aggregates trade signals from a pool of specialised RL agents.

    Agents are registered with a name, role, and initial weight.  Each
    tick they submit :class:`AgentSignal` objects.  The coordinator then
    aggregates these into a single integer action.
    """

    def __init__(self) -> None:
        self._agents: dict[str, _AgentMeta] = {}
        self._pending_signals: list[AgentSignal] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(
        self, agent_id: str, role: AgentRole, weight: float = 1.0
    ) -> None:
        """Register a new agent.  Idempotent — re-registration updates weight."""
        self._agents[agent_id] = _AgentMeta(role=role, weight=max(weight, 0.0))

    # ------------------------------------------------------------------
    # Signal submission
    # ------------------------------------------------------------------

    def submit_signal(self, signal: AgentSignal) -> None:
        """Accept a signal from a registered agent."""
        self._pending_signals.append(signal)

    def _current_signals(self) -> list[AgentSignal]:
        """Return the most recent signal per agent (last wins)."""
        latest: dict[str, AgentSignal] = {}
        for sig in self._pending_signals:
            latest[sig.agent_id] = sig
        return list(latest.values())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_signals(self, method: str = "weighted_vote") -> int:
        """Combine all pending signals into a single action.

        Args:
            method: One of ``"weighted_vote"``, ``"majority"``,
                    ``"confidence_weighted"``.

        Returns:
            +1 (buy), −1 (sell), or 0 (hold).
        """
        signals = self._current_signals()
        if not signals:
            return 0

        if method == "majority":
            return self._majority_vote(signals)
        elif method == "confidence_weighted":
            return self._confidence_weighted(signals)
        else:  # weighted_vote (default)
            return self._weighted_vote(signals)

    def _majority_vote(self, signals: list[AgentSignal]) -> int:
        counts: dict[int, int] = {1: 0, -1: 0, 0: 0}
        for sig in signals:
            counts[sig.action] = counts.get(sig.action, 0) + 1
        return max(counts, key=lambda a: counts[a])

    def _weighted_vote(self, signals: list[AgentSignal]) -> int:
        score: dict[int, float] = {1: 0.0, -1: 0.0, 0: 0.0}
        for sig in signals:
            w = self._agents.get(sig.agent_id, _AgentMeta(sig.role, 1.0)).weight
            score[sig.action] = score.get(sig.action, 0.0) + w
        return max(score, key=lambda a: score[a])

    def _confidence_weighted(self, signals: list[AgentSignal]) -> int:
        score: dict[int, float] = {1: 0.0, -1: 0.0, 0: 0.0}
        for sig in signals:
            w = self._agents.get(sig.agent_id, _AgentMeta(sig.role, 1.0)).weight
            score[sig.action] = score.get(sig.action, 0.0) + w * sig.confidence
        return max(score, key=lambda a: score[a])

    # ------------------------------------------------------------------
    # Weight adaptation
    # ------------------------------------------------------------------

    def update_weights(self, agent_id: str, realized_return: float) -> None:
        """Increase agent weight for positive realized return, penalize negative.

        Uses multiplicative update: weight *= (1 + 0.1 * sign(return)).
        """
        meta = self._agents.get(agent_id)
        if meta is None:
            return
        if realized_return > 0:
            meta.weight *= 1.1
            meta.correct_calls += 1
        elif realized_return < 0:
            meta.weight *= 0.9
        meta.total_calls += 1
        # Keep weight bounded away from zero.
        meta.weight = max(meta.weight, 0.01)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def conflict_score(self) -> float:
        """Measure agent disagreement.

        Returns a value in [0.0, 1.0]:
        - 0.0 = all agents agree unanimously.
        - 1.0 = agents split equally across all three actions.
        """
        signals = self._current_signals()
        if not signals:
            return 0.0

        counts: dict[int, int] = {}
        for sig in signals:
            counts[sig.action] = counts.get(sig.action, 0) + 1

        n = len(signals)
        # Normalized Herfindahl-Hirschman index of disagreement.
        # HHI = 1.0 when unanimous -> conflict = 0.0
        # HHI = 1/k when maximally split -> conflict = 1.0
        hhi = sum((c / n) ** 2 for c in counts.values())
        max_k = min(len(counts), 3)
        min_hhi = 1.0 / max_k if max_k > 0 else 1.0
        if hhi >= 1.0:
            # Unanimous
            return 0.0
        if hhi <= min_hhi:
            return 1.0
        return max(0.0, 1.0 - (hhi - min_hhi) / (1.0 - min_hhi))

    def dominant_role(self) -> Optional[AgentRole]:
        """Return the AgentRole with the highest cumulative weight.

        Returns *None* if no agents are registered.
        """
        if not self._agents:
            return None
        role_weight: dict[AgentRole, float] = {}
        for meta in self._agents.values():
            role_weight[meta.role] = role_weight.get(meta.role, 0.0) + meta.weight
        return max(role_weight, key=lambda r: role_weight[r])


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestAgentRole(unittest.TestCase):
    def test_enum_values(self) -> None:
        self.assertIn(AgentRole.TREND_FOLLOWER, list(AgentRole))
        self.assertIn(AgentRole.MEAN_REVERTER, list(AgentRole))
        self.assertIn(AgentRole.MOMENTUM, list(AgentRole))
        self.assertIn(AgentRole.VOLATILITY_TRADER, list(AgentRole))

    def test_enum_count(self) -> None:
        self.assertEqual(len(AgentRole), 4)


class TestMultiAgentCoordinator(unittest.TestCase):
    def _make_signal(
        self,
        agent_id: str,
        action: int,
        confidence: float = 1.0,
        role: AgentRole = AgentRole.TREND_FOLLOWER,
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=agent_id,
            role=role,
            action=action,
            confidence=confidence,
        )

    def test_register_agent(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", AgentRole.TREND_FOLLOWER, weight=1.5)
        self.assertIn("a1", coord._agents)

    def test_submit_signal_stored(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", AgentRole.MOMENTUM)
        coord.submit_signal(self._make_signal("a1", 1))
        self.assertEqual(len(coord._pending_signals), 1)

    def test_majority_vote_buy(self) -> None:
        coord = MultiAgentCoordinator()
        for aid in ["a", "b", "c"]:
            coord.register_agent(aid, AgentRole.MOMENTUM)
        coord.submit_signal(self._make_signal("a", 1))
        coord.submit_signal(self._make_signal("b", 1))
        coord.submit_signal(self._make_signal("c", -1))
        self.assertEqual(coord.aggregate_signals("majority"), 1)

    def test_weighted_vote_higher_weight_wins(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("heavy", AgentRole.TREND_FOLLOWER, weight=10.0)
        coord.register_agent("light", AgentRole.MEAN_REVERTER, weight=0.1)
        coord.submit_signal(self._make_signal("heavy", -1))
        coord.submit_signal(self._make_signal("light", 1))
        self.assertEqual(coord.aggregate_signals("weighted_vote"), -1)

    def test_confidence_weighted_aggregation(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a", AgentRole.MOMENTUM, weight=1.0)
        coord.register_agent("b", AgentRole.MOMENTUM, weight=1.0)
        coord.submit_signal(self._make_signal("a", 1, confidence=0.9))
        coord.submit_signal(self._make_signal("b", -1, confidence=0.1))
        result = coord.aggregate_signals("confidence_weighted")
        self.assertEqual(result, 1)

    def test_no_signals_returns_hold(self) -> None:
        coord = MultiAgentCoordinator()
        self.assertEqual(coord.aggregate_signals(), 0)

    def test_update_weights_increases_on_profit(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", AgentRole.TREND_FOLLOWER, weight=1.0)
        coord.update_weights("a1", 0.05)
        self.assertGreater(coord._agents["a1"].weight, 1.0)

    def test_update_weights_decreases_on_loss(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", AgentRole.TREND_FOLLOWER, weight=1.0)
        coord.update_weights("a1", -0.03)
        self.assertLess(coord._agents["a1"].weight, 1.0)

    def test_conflict_score_unanimous(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a", AgentRole.MOMENTUM)
        coord.register_agent("b", AgentRole.MOMENTUM)
        coord.submit_signal(self._make_signal("a", 1))
        coord.submit_signal(self._make_signal("b", 1))
        self.assertAlmostEqual(coord.conflict_score(), 0.0)

    def test_conflict_score_split(self) -> None:
        coord = MultiAgentCoordinator()
        for aid in ["a", "b"]:
            coord.register_agent(aid, AgentRole.MOMENTUM)
        coord.submit_signal(self._make_signal("a", 1))
        coord.submit_signal(self._make_signal("b", -1))
        # Some non-zero conflict.
        self.assertGreater(coord.conflict_score(), 0.0)

    def test_dominant_role_none_when_empty(self) -> None:
        coord = MultiAgentCoordinator()
        self.assertIsNone(coord.dominant_role())

    def test_dominant_role_returns_highest_weight(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a", AgentRole.TREND_FOLLOWER, weight=5.0)
        coord.register_agent("b", AgentRole.MEAN_REVERTER, weight=1.0)
        self.assertEqual(coord.dominant_role(), AgentRole.TREND_FOLLOWER)

    def test_update_unknown_agent_noop(self) -> None:
        coord = MultiAgentCoordinator()
        coord.update_weights("ghost", 1.0)  # should not raise

    def test_last_signal_per_agent_used(self) -> None:
        coord = MultiAgentCoordinator()
        coord.register_agent("a", AgentRole.MOMENTUM)
        coord.submit_signal(self._make_signal("a", -1))
        coord.submit_signal(self._make_signal("a", 1))
        self.assertEqual(coord.aggregate_signals("majority"), 1)


if __name__ == "__main__":
    unittest.main()

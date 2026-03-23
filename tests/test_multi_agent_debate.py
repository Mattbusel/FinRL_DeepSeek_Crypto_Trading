"""Tests for multi_agent_debate.py."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multi_agent_debate import (
    AgentDebate,
    AgentRole,
    BearAgent,
    BullAgent,
    DebatePosition,
    DebateRound,
    DebateVerdict,
    DeepSeekDebater,
    NeutralAgent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call_fn(position: str, evidence: str = "Some evidence", confidence: float = 0.8):
    """Return a mock call_fn that always succeeds with the given payload."""
    payload = json.dumps(
        {
            "position_argued": position,
            "evidence": evidence,
            "counter_argument": "Counter point here",
            "confidence": confidence,
        }
    )

    def call_fn(prompt: str):
        return payload, json.loads(payload)

    return call_fn


def _make_debater_with_mocked_api(
    n_rounds: int = 1,
    bull_confidence: float = 0.75,
    bear_confidence: float = 0.65,
    neutral_position: str = "HOLD",
    arbiter_action: str = "LONG",
) -> tuple[DeepSeekDebater, MagicMock]:
    """Build a DeepSeekDebater whose _call method is fully mocked."""
    debater = DeepSeekDebater.__new__(DeepSeekDebater)
    debater._api_key = "fake-key"
    debater._model = "deepseek-chat"
    debater._temperature = 0.3
    debater._max_tokens = 512
    debater._client = MagicMock()

    call_responses = []

    for _ in range(n_rounds):
        # Bull response
        call_responses.append(
            (
                json.dumps(
                    {
                        "position_argued": "LONG",
                        "evidence": "Bull evidence",
                        "counter_argument": "Bull counter",
                        "confidence": bull_confidence,
                    }
                ),
                {
                    "position_argued": "LONG",
                    "evidence": "Bull evidence",
                    "counter_argument": "Bull counter",
                    "confidence": bull_confidence,
                },
            )
        )
        # Bear response
        call_responses.append(
            (
                json.dumps(
                    {
                        "position_argued": "SHORT",
                        "evidence": "Bear evidence",
                        "counter_argument": "Bear counter",
                        "confidence": bear_confidence,
                    }
                ),
                {
                    "position_argued": "SHORT",
                    "evidence": "Bear evidence",
                    "counter_argument": "Bear counter",
                    "confidence": bear_confidence,
                },
            )
        )
        # Neutral response
        call_responses.append(
            (
                json.dumps(
                    {
                        "position_argued": neutral_position,
                        "evidence": "Neutral evidence",
                        "counter_argument": "Neutral counter",
                        "confidence": 0.6,
                    }
                ),
                {
                    "position_argued": neutral_position,
                    "evidence": "Neutral evidence",
                    "counter_argument": "Neutral counter",
                    "confidence": 0.6,
                },
            )
        )

    # Arbiter response
    call_responses.append(
        (
            json.dumps(
                {
                    "final_action": arbiter_action,
                    "consensus_confidence": 0.7,
                    "key_bull_point": "Accumulation signal",
                    "key_bear_point": "RSI overbought",
                    "rationale": "Bulls outweigh bears.",
                }
            ),
            {
                "final_action": arbiter_action,
                "consensus_confidence": 0.7,
                "key_bull_point": "Accumulation signal",
                "key_bear_point": "RSI overbought",
                "rationale": "Bulls outweigh bears.",
            },
        )
    )

    call_mock = MagicMock(side_effect=call_responses)
    debater._call = call_mock

    # Attach fresh sub-agents using the mocked _call
    from multi_agent_debate import BearAgent, BullAgent, NeutralAgent

    debater._bull = BullAgent(call_fn=debater._call)
    debater._bear = BearAgent(call_fn=debater._call)
    debater._neutral = NeutralAgent(call_fn=debater._call)

    return debater, call_mock


# ---------------------------------------------------------------------------
# DebateRound
# ---------------------------------------------------------------------------


class TestDebateRound:
    def test_fields_stored(self):
        dr = DebateRound(
            round_number=1,
            agent_role=AgentRole.BULL,
            position_argued=DebatePosition.LONG,
            evidence="Strong on-chain data",
            counter_argument="N/A",
            confidence=0.85,
        )
        assert dr.round_number == 1
        assert dr.agent_role == AgentRole.BULL
        assert dr.position_argued == DebatePosition.LONG
        assert abs(dr.confidence - 0.85) < 1e-6

    def test_default_raw_response(self):
        dr = DebateRound(
            round_number=1,
            agent_role=AgentRole.BEAR,
            position_argued=DebatePosition.SHORT,
            evidence="High funding rates",
            counter_argument="Bull ignores sell pressure",
            confidence=0.7,
        )
        assert dr.raw_response == ""


# ---------------------------------------------------------------------------
# DebateVerdict
# ---------------------------------------------------------------------------


class TestDebateVerdict:
    def _make_verdict(self, action: DebatePosition = DebatePosition.LONG) -> DebateVerdict:
        bull_r = DebateRound(1, AgentRole.BULL, DebatePosition.LONG, "e1", "c1", 0.8)
        bear_r = DebateRound(1, AgentRole.BEAR, DebatePosition.SHORT, "e2", "c2", 0.7)
        neutral_r = DebateRound(1, AgentRole.NEUTRAL, action, "e3", "c3", 0.6)
        return DebateVerdict(
            final_action=action,
            consensus_confidence=0.7,
            bull_rounds=[bull_r],
            bear_rounds=[bear_r],
            neutral_rounds=[neutral_r],
            reasoning_chain=[("BULL R1", "evidence"), ("ARBITER", "rationale")],
        )

    def test_all_rounds_sorted(self):
        verdict = self._make_verdict()
        all_r = verdict.all_rounds
        assert len(all_r) == 3
        # Should be sorted by (round_number, role)
        for r in all_r:
            assert r.round_number == 1

    def test_summary_contains_action(self):
        verdict = self._make_verdict(DebatePosition.LONG)
        s = verdict.summary()
        assert "LONG" in s
        assert "Confidence" in s
        assert "Reasoning chain" in s

    def test_dissent_flag(self):
        verdict = self._make_verdict(DebatePosition.LONG)
        verdict.dissent = True
        assert "DISSENT" in verdict.summary()


# ---------------------------------------------------------------------------
# BullAgent
# ---------------------------------------------------------------------------


class TestBullAgent:
    def test_always_argues_long(self):
        call_fn = _make_call_fn("SHORT")  # LLM returns SHORT
        agent = BullAgent(call_fn=call_fn)
        dr = agent.argue("BTC", "Bullish context", round_number=1)
        # BullAgent must override to LONG regardless
        assert dr.position_argued == DebatePosition.LONG

    def test_confidence_captured(self):
        call_fn = _make_call_fn("LONG", confidence=0.92)
        agent = BullAgent(call_fn=call_fn)
        dr = agent.argue("BTC", "Context", round_number=1)
        assert abs(dr.confidence - 0.92) < 1e-6

    def test_round_number_stored(self):
        call_fn = _make_call_fn("LONG")
        agent = BullAgent(call_fn=call_fn)
        dr = agent.argue("BTC", "Context", round_number=3)
        assert dr.round_number == 3

    def test_agent_role(self):
        call_fn = _make_call_fn("LONG")
        agent = BullAgent(call_fn=call_fn)
        dr = agent.argue("ETH", "Context", round_number=1)
        assert dr.agent_role == AgentRole.BULL


# ---------------------------------------------------------------------------
# BearAgent
# ---------------------------------------------------------------------------


class TestBearAgent:
    def test_always_argues_short(self):
        call_fn = _make_call_fn("LONG")  # LLM returns LONG
        agent = BearAgent(call_fn=call_fn)
        dr = agent.argue("BTC", "Bearish context", round_number=1)
        assert dr.position_argued == DebatePosition.SHORT

    def test_evidence_captured(self):
        evidence = "RSI at 80, funding rates spiked"
        call_fn = _make_call_fn("SHORT", evidence=evidence)
        agent = BearAgent(call_fn=call_fn)
        dr = agent.argue("BTC", "Context", round_number=1)
        assert dr.evidence == evidence


# ---------------------------------------------------------------------------
# NeutralAgent
# ---------------------------------------------------------------------------


class TestNeutralAgent:
    def test_hold_verdict(self):
        call_fn = _make_call_fn("HOLD")
        agent = NeutralAgent(call_fn=call_fn)
        bull_r = DebateRound(1, AgentRole.BULL, DebatePosition.LONG, "bull ev", "c", 0.8)
        bear_r = DebateRound(1, AgentRole.BEAR, DebatePosition.SHORT, "bear ev", "c", 0.7)
        dr = agent.moderate("BTC", "Context", 1, bull_r, bear_r)
        assert dr.position_argued == DebatePosition.HOLD
        assert dr.agent_role == AgentRole.NEUTRAL

    def test_long_verdict(self):
        call_fn = _make_call_fn("LONG")
        agent = NeutralAgent(call_fn=call_fn)
        bull_r = DebateRound(1, AgentRole.BULL, DebatePosition.LONG, "ev", "c", 0.9)
        bear_r = DebateRound(1, AgentRole.BEAR, DebatePosition.SHORT, "ev", "c", 0.5)
        dr = agent.moderate("BTC", "Context", 1, bull_r, bear_r)
        assert dr.position_argued == DebatePosition.LONG

    def test_invalid_position_fallback(self):
        call_fn = _make_call_fn("UNKNOWN_POSITION")
        agent = NeutralAgent(call_fn=call_fn)
        bull_r = DebateRound(1, AgentRole.BULL, DebatePosition.LONG, "ev", "c", 0.8)
        bear_r = DebateRound(1, AgentRole.BEAR, DebatePosition.SHORT, "ev", "c", 0.7)
        dr = agent.moderate("BTC", "Context", 1, bull_r, bear_r)
        assert dr.position_argued == DebatePosition.HOLD  # fallback


# ---------------------------------------------------------------------------
# DeepSeekDebater
# ---------------------------------------------------------------------------


class TestDeepSeekDebater:
    def test_run_debate_1_round_long(self):
        debater, _ = _make_debater_with_mocked_api(n_rounds=1, arbiter_action="LONG")
        verdict = debater.run_debate("BTC", "Strong bull market", n_rounds=1)
        assert isinstance(verdict, DebateVerdict)
        assert verdict.final_action == DebatePosition.LONG
        assert len(verdict.bull_rounds) == 1
        assert len(verdict.bear_rounds) == 1
        assert len(verdict.neutral_rounds) == 1

    def test_run_debate_2_rounds(self):
        debater, _ = _make_debater_with_mocked_api(n_rounds=2, arbiter_action="SHORT")
        verdict = debater.run_debate("ETH", "Uncertain market", n_rounds=2)
        assert len(verdict.bull_rounds) == 2
        assert len(verdict.bear_rounds) == 2
        assert len(verdict.neutral_rounds) == 2

    def test_consensus_confidence_range(self):
        debater, _ = _make_debater_with_mocked_api(n_rounds=1)
        verdict = debater.run_debate("BTC", "Context", n_rounds=1)
        assert 0.0 <= verdict.consensus_confidence <= 1.0

    def test_reasoning_chain_nonempty(self):
        debater, _ = _make_debater_with_mocked_api(n_rounds=1)
        verdict = debater.run_debate("BTC", "Context", n_rounds=1)
        assert len(verdict.reasoning_chain) > 0

    def test_dissent_detected(self):
        # Neutral says HOLD but arbiter says LONG -> dissent
        debater, _ = _make_debater_with_mocked_api(
            n_rounds=1, neutral_position="HOLD", arbiter_action="LONG"
        )
        verdict = debater.run_debate("BTC", "Context", n_rounds=1)
        assert verdict.dissent is True

    def test_no_dissent_when_aligned(self):
        # Neutral says LONG and arbiter says LONG -> no dissent
        debater, _ = _make_debater_with_mocked_api(
            n_rounds=1, neutral_position="LONG", arbiter_action="LONG"
        )
        verdict = debater.run_debate("BTC", "Context", n_rounds=1)
        assert verdict.dissent is False

    def test_invalid_n_rounds_raises(self):
        debater, _ = _make_debater_with_mocked_api()
        with pytest.raises(ValueError, match="n_rounds"):
            debater.run_debate("BTC", "Context", n_rounds=0)

    def test_missing_api_key_raises(self):
        from exceptions import SignalError

        debater = DeepSeekDebater(api_key="")
        with pytest.raises(SignalError, match="DEEPSEEK_API_KEY"):
            debater._get_client()


# ---------------------------------------------------------------------------
# AgentDebate (facade)
# ---------------------------------------------------------------------------


class TestAgentDebate:
    def test_decide_returns_verdict(self):
        debater_mock, _ = _make_debater_with_mocked_api(n_rounds=1)
        debate = AgentDebate(n_rounds=1, debater=debater_mock)
        verdict = debate.decide("BTC", "Context")
        assert isinstance(verdict, DebateVerdict)

    def test_override_n_rounds(self):
        debater_mock, _ = _make_debater_with_mocked_api(n_rounds=2)
        debate = AgentDebate(n_rounds=1, debater=debater_mock)
        verdict = debate.decide("BTC", "Context", n_rounds=2)
        assert len(verdict.bull_rounds) == 2

"""Multi-agent debate system for trade decision making.

Three specialised agents with conflicting priors debate a proposed trade over
N rounds.  A ``NeutralAgent`` moderates the debate, identifying logical flaws.
After all rounds, ``DebateVerdict`` encodes the consensus decision together
with the full reasoning chain.

``DeepSeekDebater`` drives each agent's arguments through the DeepSeek V3 API
so every position is grounded in LLM reasoning rather than heuristics.

Typical usage::

    from multi_agent_debate import AgentDebate, DeepSeekDebater

    debater = DeepSeekDebater()
    verdict = debater.run_debate(
        asset="BTC",
        market_context="BTC just broke above $70k on strong volume; RSI=72.",
        n_rounds=2,
    )
    print(verdict.final_action)  # "LONG" | "SHORT" | "HOLD"
    print(verdict.consensus_confidence)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import OpenAI

from config import settings
from exceptions import SignalError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DebatePosition(str, Enum):
    """Stance taken by a debating agent in a single round."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class AgentRole(str, Enum):
    """Functional role of a debating agent."""

    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DebateRound:
    """One round of debate contributed by a single agent.

    Attributes:
        round_number: 1-based index of this round.
        agent_role: Which agent produced this round.
        position_argued: The trade position the agent is arguing for.
        evidence: Key facts or indicators the agent cites.
        counter_argument: Rebuttal of the *opposing* agent's previous argument.
        confidence: Self-assessed confidence in the argument (0.0 – 1.0).
        raw_response: Raw JSON string returned by the LLM.
    """

    round_number: int
    agent_role: AgentRole
    position_argued: DebatePosition
    evidence: str
    counter_argument: str
    confidence: float
    raw_response: str = ""


@dataclass
class DebateVerdict:
    """Consensus decision produced after all debate rounds complete.

    Attributes:
        final_action: The agreed trade action (LONG / SHORT / HOLD).
        consensus_confidence: Weighted average confidence across all rounds.
        bull_rounds: All rounds contributed by the BullAgent.
        bear_rounds: All rounds contributed by the BearAgent.
        neutral_rounds: All rounds contributed by the NeutralAgent.
        reasoning_chain: Ordered list of (role, summary) pairs for the full
            reasoning trail.
        dissent: True when agents did not reach agreement; the moderator's
            tie-breaking decision is used.
    """

    final_action: DebatePosition
    consensus_confidence: float
    bull_rounds: list[DebateRound]
    bear_rounds: list[DebateRound]
    neutral_rounds: list[DebateRound]
    reasoning_chain: list[tuple[str, str]] = field(default_factory=list)
    dissent: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def all_rounds(self) -> list[DebateRound]:
        """All rounds in chronological order (interleaved by round number)."""
        combined = self.bull_rounds + self.bear_rounds + self.neutral_rounds
        return sorted(combined, key=lambda r: (r.round_number, r.agent_role.value))

    def summary(self) -> str:
        """Return a concise human-readable verdict summary."""
        dissent_tag = " [DISSENT - moderator tiebreak]" if self.dissent else ""
        lines = [
            f"=== Debate Verdict{dissent_tag} ===",
            f"Final action : {self.final_action.value}",
            f"Confidence   : {self.consensus_confidence:.2%}",
            "",
            "Reasoning chain:",
        ]
        for role, text in self.reasoning_chain:
            lines.append(f"  [{role}] {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BULL_PROMPT = """\
You are BullAgent — an aggressive cryptocurrency analyst who always looks for \
reasons to go LONG on {asset}.

Market context: {context}

Previous bear argument (round {round_num}): {bear_prev}

Your task (respond ONLY with a JSON object):
- "position_argued": always "LONG"
- "evidence": 2-3 concrete bullish indicators or catalysts you see
- "counter_argument": one clear logical flaw in the bear's previous argument \
(or "N/A" if round 1)
- "confidence": float 0.0–1.0 representing your conviction

Example:
{{"position_argued":"LONG","evidence":"RSI reset from oversold; on-chain \
accumulation rising; macro tailwind from rate-cut expectations","counter_argument":\
"Bear ignores that the regulatory crackdown is already priced in","confidence":0.78}}
"""

_BEAR_PROMPT = """\
You are BearAgent — a cautious cryptocurrency analyst who always looks for \
reasons to go SHORT or exit on {asset}.

Market context: {context}

Previous bull argument (round {round_num}): {bull_prev}

Your task (respond ONLY with a JSON object):
- "position_argued": always "SHORT"
- "evidence": 2-3 concrete bearish indicators or risks you see
- "counter_argument": one clear logical flaw in the bull's previous argument \
(or "N/A" if round 1)
- "confidence": float 0.0–1.0 representing your conviction

Example:
{{"position_argued":"SHORT","evidence":"RSI overbought at 78; funding rates \
spiking; distribution pattern on-chain","counter_argument":"Bull overlooks \
that ETF inflows have slowed 40% week-over-week","confidence":0.72}}
"""

_NEUTRAL_PROMPT = """\
You are NeutralAgent — an impartial debate moderator for {asset} trading decisions.

Market context: {context}

Bull's latest argument  : {bull_latest}
Bear's latest argument  : {bear_latest}

Your task (respond ONLY with a JSON object):
- "position_argued": one of "LONG", "SHORT", or "HOLD" — your reasoned verdict \
after weighing both sides
- "evidence": the single strongest piece of evidence from either side
- "counter_argument": the single most significant logical flaw you identified \
across both arguments
- "confidence": float 0.0–1.0 for your verdict

Example:
{{"position_argued":"HOLD","evidence":"On-chain accumulation is genuinely \
bullish but RSI divergence adds meaningful downside risk","counter_argument":\
"Both agents ignore that macro uncertainty makes short-term directional bets \
high-risk","confidence":0.61}}
"""

_VERDICT_PROMPT = """\
You are the Chief Arbiter for a multi-agent trading debate on {asset}.

Full debate transcript (JSON array of rounds):
{transcript}

Based solely on the evidence and rebuttals in the transcript, produce a final \
verdict as a JSON object:
- "final_action": one of "LONG", "SHORT", or "HOLD"
- "consensus_confidence": float 0.0–1.0 — overall confidence considering \
agreement/disagreement across agents
- "key_bull_point": the strongest bullish argument from the debate
- "key_bear_point": the strongest bearish argument from the debate
- "rationale": one sentence explaining the final decision

Example:
{{"final_action":"LONG","consensus_confidence":0.69,"key_bull_point":"On-chain \
accumulation signals institutional buying","key_bear_point":"Overbought RSI \
raises short-term reversal risk","rationale":"Accumulation evidence outweighs \
short-term technical risk given current macro tailwinds"}}
"""


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------


class _BaseDebateAgent:
    """Abstract base for all three debating agents.

    Args:
        role: The :class:`AgentRole` this agent represents.
        call_fn: Callable that accepts a prompt str and returns a parsed JSON
            dict.  Injected by :class:`DeepSeekDebater` to allow easy mocking
            in tests.
    """

    def __init__(self, role: AgentRole, call_fn: Any) -> None:
        self.role = role
        self._call = call_fn

    def _parse_round(self, data: dict[str, Any], round_number: int, raw: str) -> DebateRound:
        """Convert a raw API response dict into a :class:`DebateRound`.

        Falls back to safe defaults for any missing fields so the debate can
        continue even if the LLM omits a key.
        """
        position_raw = str(data.get("position_argued", self.role.value)).upper()
        try:
            position = DebatePosition(position_raw)
        except ValueError:
            position = DebatePosition.HOLD

        return DebateRound(
            round_number=round_number,
            agent_role=self.role,
            position_argued=position,
            evidence=str(data.get("evidence", "")),
            counter_argument=str(data.get("counter_argument", "N/A")),
            confidence=float(data.get("confidence", 0.5)),
            raw_response=raw,
        )


class BullAgent(_BaseDebateAgent):
    """Argues exclusively for LONG positions.

    Args:
        call_fn: LLM call function (prompt -> dict).
    """

    def __init__(self, call_fn: Any) -> None:
        super().__init__(AgentRole.BULL, call_fn)

    def argue(
        self,
        asset: str,
        context: str,
        round_number: int,
        bear_prev_argument: str = "N/A",
    ) -> DebateRound:
        """Generate a bullish argument for *round_number*.

        Args:
            asset: Ticker symbol, e.g. ``"BTC"``.
            context: Narrative market context string.
            round_number: 1-based debate round index.
            bear_prev_argument: The bear's argument from the previous round.

        Returns:
            A :class:`DebateRound` with LONG position.
        """
        prompt = _BULL_PROMPT.format(
            asset=asset,
            context=context,
            round_num=round_number,
            bear_prev=bear_prev_argument,
        )
        raw_str, data = self._call(prompt)
        dr = self._parse_round(data, round_number, raw_str)
        # BullAgent always argues LONG regardless of LLM override
        dr.position_argued = DebatePosition.LONG
        log.debug(
            "bull.argued",
            round=round_number,
            confidence=dr.confidence,
            evidence=dr.evidence[:80],
        )
        return dr


class BearAgent(_BaseDebateAgent):
    """Argues exclusively for SHORT positions.

    Args:
        call_fn: LLM call function (prompt -> dict).
    """

    def __init__(self, call_fn: Any) -> None:
        super().__init__(AgentRole.BEAR, call_fn)

    def argue(
        self,
        asset: str,
        context: str,
        round_number: int,
        bull_prev_argument: str = "N/A",
    ) -> DebateRound:
        """Generate a bearish argument for *round_number*.

        Args:
            asset: Ticker symbol.
            context: Narrative market context string.
            round_number: 1-based debate round index.
            bull_prev_argument: The bull's argument from the previous round.

        Returns:
            A :class:`DebateRound` with SHORT position.
        """
        prompt = _BEAR_PROMPT.format(
            asset=asset,
            context=context,
            round_num=round_number,
            bull_prev=bull_prev_argument,
        )
        raw_str, data = self._call(prompt)
        dr = self._parse_round(data, round_number, raw_str)
        dr.position_argued = DebatePosition.SHORT
        log.debug(
            "bear.argued",
            round=round_number,
            confidence=dr.confidence,
            evidence=dr.evidence[:80],
        )
        return dr


class NeutralAgent(_BaseDebateAgent):
    """Moderates the debate, identifies logical flaws, and provides a balanced verdict.

    Args:
        call_fn: LLM call function (prompt -> dict).
    """

    def __init__(self, call_fn: Any) -> None:
        super().__init__(AgentRole.NEUTRAL, call_fn)

    def moderate(
        self,
        asset: str,
        context: str,
        round_number: int,
        bull_round: DebateRound,
        bear_round: DebateRound,
    ) -> DebateRound:
        """Evaluate the bull and bear arguments for *round_number*.

        Args:
            asset: Ticker symbol.
            context: Narrative market context string.
            round_number: 1-based debate round index.
            bull_round: Bull's :class:`DebateRound` for this round.
            bear_round: Bear's :class:`DebateRound` for this round.

        Returns:
            A :class:`DebateRound` with an impartial LONG / SHORT / HOLD verdict.
        """
        prompt = _NEUTRAL_PROMPT.format(
            asset=asset,
            context=context,
            bull_latest=bull_round.evidence,
            bear_latest=bear_round.evidence,
        )
        raw_str, data = self._call(prompt)
        dr = self._parse_round(data, round_number, raw_str)
        log.debug(
            "neutral.moderated",
            round=round_number,
            verdict=dr.position_argued.value,
            confidence=dr.confidence,
        )
        return dr


# ---------------------------------------------------------------------------
# DeepSeekDebater — orchestrator
# ---------------------------------------------------------------------------


class DeepSeekDebater:
    """Orchestrates a multi-round debate using the DeepSeek V3 API.

    Each of the three agents (Bull, Bear, Neutral) is called in sequence for
    every round.  After all rounds a Chief Arbiter prompt produces the final
    :class:`DebateVerdict`.

    Args:
        api_key: DeepSeek API key (defaults to ``settings.deepseek_api_key``).
        model: Model ID (defaults to ``settings.deepseek_model``).
        temperature: Sampling temperature (default 0.3 for some diversity).
        max_tokens: Max completion tokens per call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> None:
        self._api_key = api_key or settings.deepseek_api_key
        self._model = model or settings.deepseek_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: OpenAI | None = None

        self._bull = BullAgent(call_fn=self._call)
        self._bear = BearAgent(call_fn=self._call)
        self._neutral = NeutralAgent(call_fn=self._call)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._api_key:
                raise SignalError(
                    "DEEPSEEK_API_KEY is not set. Cannot run multi-agent debate."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=settings.deepseek_base_url,
            )
        return self._client

    def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Send *prompt* to the DeepSeek API; return (raw_str, parsed_dict).

        Retries up to ``settings.max_retries`` times with exponential backoff.

        Returns:
            A tuple of (raw JSON string, parsed dict).

        Raises:
            SignalError: After all retries are exhausted.
        """
        backoff = list(settings.retry_backoff_seconds)
        max_retries = settings.max_retries
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                return raw, json.loads(raw)
            except Exception as exc:
                last_exc = exc
                wait = backoff[min(attempt, len(backoff) - 1)] + random.uniform(0, 1)
                log.warning(
                    "debate.api_retry",
                    attempt=attempt + 1,
                    wait=round(wait, 2),
                    error=str(exc),
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        raise SignalError(
            f"DeepSeek API failed after {max_retries} attempts.",
            cause=last_exc,
        )

    def _call_verdict(
        self,
        asset: str,
        all_rounds: list[DebateRound],
    ) -> dict[str, Any]:
        """Call the Chief Arbiter prompt to get the final verdict dict."""
        transcript = json.dumps(
            [
                {
                    "round": r.round_number,
                    "agent": r.agent_role.value,
                    "position": r.position_argued.value,
                    "evidence": r.evidence,
                    "counter": r.counter_argument,
                    "confidence": r.confidence,
                }
                for r in all_rounds
            ],
            indent=2,
        )
        prompt = _VERDICT_PROMPT.format(asset=asset, transcript=transcript)
        _, data = self._call(prompt)
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_debate(
        self,
        asset: str,
        market_context: str,
        n_rounds: int = 2,
    ) -> DebateVerdict:
        """Run a full N-round debate and return a :class:`DebateVerdict`.

        Args:
            asset: Ticker symbol (e.g. ``"BTC"``).
            market_context: Free-text description of current market conditions.
            n_rounds: Number of full debate rounds (each round includes one
                Bull, one Bear, and one Neutral call).  Minimum 1.

        Returns:
            A :class:`DebateVerdict` with the final trade decision and full
            reasoning chain.

        Raises:
            SignalError: If the API is unreachable after retries.
            ValueError: If *n_rounds* < 1.
        """
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

        log.info(
            "debate.start",
            asset=asset,
            n_rounds=n_rounds,
            context=market_context[:100],
        )

        bull_rounds: list[DebateRound] = []
        bear_rounds: list[DebateRound] = []
        neutral_rounds: list[DebateRound] = []

        bear_prev = "N/A"
        bull_prev = "N/A"

        for rn in range(1, n_rounds + 1):
            log.info("debate.round", round=rn, total=n_rounds)

            bull_dr = self._bull.argue(asset, market_context, rn, bear_prev)
            bear_dr = self._bear.argue(asset, market_context, rn, bull_prev)
            neutral_dr = self._neutral.moderate(asset, market_context, rn, bull_dr, bear_dr)

            bull_rounds.append(bull_dr)
            bear_rounds.append(bear_dr)
            neutral_rounds.append(neutral_dr)

            bull_prev = bull_dr.evidence
            bear_prev = bear_dr.evidence

        # Build final verdict via Chief Arbiter
        all_rounds = bull_rounds + bear_rounds + neutral_rounds
        verdict_data = self._call_verdict(asset, all_rounds)

        final_action_raw = str(verdict_data.get("final_action", "HOLD")).upper()
        try:
            final_action = DebatePosition(final_action_raw)
        except ValueError:
            final_action = DebatePosition.HOLD

        consensus_conf = float(verdict_data.get("consensus_confidence", 0.5))

        # Check for dissent: neutral rounds disagree with Chief Arbiter
        neutral_positions = {nr.position_argued for nr in neutral_rounds}
        dissent = final_action not in neutral_positions and len(neutral_positions) > 0

        # Build reasoning chain
        reasoning_chain: list[tuple[str, str]] = []
        for dr in sorted(all_rounds, key=lambda r: (r.round_number, r.agent_role.value)):
            reasoning_chain.append(
                (
                    f"{dr.agent_role.value} R{dr.round_number}",
                    f"[{dr.position_argued.value} conf={dr.confidence:.2f}] "
                    f"{dr.evidence[:120]}",
                )
            )
        reasoning_chain.append(
            (
                "ARBITER",
                f"[{final_action.value}] "
                + str(verdict_data.get("rationale", "")),
            )
        )

        verdict = DebateVerdict(
            final_action=final_action,
            consensus_confidence=consensus_conf,
            bull_rounds=bull_rounds,
            bear_rounds=bear_rounds,
            neutral_rounds=neutral_rounds,
            reasoning_chain=reasoning_chain,
            dissent=dissent,
        )

        log.info(
            "debate.complete",
            asset=asset,
            action=final_action.value,
            confidence=round(consensus_conf, 3),
            dissent=dissent,
        )
        return verdict


# ---------------------------------------------------------------------------
# Convenience wrapper: AgentDebate
# ---------------------------------------------------------------------------


class AgentDebate:
    """High-level façade that wires up :class:`DeepSeekDebater` with sane defaults.

    Args:
        n_rounds: Default number of debate rounds (can be overridden per call).
        debater: Optional pre-configured :class:`DeepSeekDebater`; created
            automatically if omitted.
    """

    def __init__(
        self,
        n_rounds: int = 2,
        debater: DeepSeekDebater | None = None,
    ) -> None:
        self.n_rounds = n_rounds
        self._debater = debater or DeepSeekDebater()

    def decide(
        self,
        asset: str,
        market_context: str,
        n_rounds: int | None = None,
    ) -> DebateVerdict:
        """Run a debate and return the final :class:`DebateVerdict`.

        Args:
            asset: Ticker symbol, e.g. ``"BTC"``.
            market_context: Free-text market conditions.
            n_rounds: Override the default round count for this call.

        Returns:
            :class:`DebateVerdict` with final decision and reasoning chain.
        """
        rounds = n_rounds if n_rounds is not None else self.n_rounds
        return self._debater.run_debate(asset, market_context, n_rounds=rounds)

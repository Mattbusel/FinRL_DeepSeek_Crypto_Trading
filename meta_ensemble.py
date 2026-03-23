"""Meta-ensemble that combines multiple RL agent types via a learned meta-learner.

Supported base agents: DQN (AgentDoubleDQN), D3QN (AgentD3QN), DDPG-style
(AgentTwinD3QN).  A lightweight online meta-learner weights their action
signals based on *recent per-regime performance*, so the meta-ensemble
adapts which sub-agent to trust as market conditions change.

Architecture
------------
1. Each base agent produces an action in ``{-1, 0, 1}`` (sell / hold / buy).
2. Performance tracker records rolling accuracy per (agent, regime) window.
3. Meta-learner converts performance scores to softmax weights.
4. Final action = sign of weighted vote (ties → 0 hold).

Example::

    from meta_ensemble import MetaEnsemble
    from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
    from erl_config import Config

    cfg = Config()
    agents = {
        "dqn":   AgentDoubleDQN(net_dims=(128, 64), state_dim=12, action_dim=3, gpu_id=-1),
        "d3qn":  AgentD3QN(net_dims=(128, 64),      state_dim=12, action_dim=3, gpu_id=-1),
        "ddpg":  AgentTwinD3QN(net_dims=(128, 64),  state_dim=12, action_dim=3, gpu_id=-1),
    }
    ensemble = MetaEnsemble(agents=agents)
    action = ensemble.act(state_tensor, regime=MarketRegime.BULL)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from logger import get_logger
from regime_detector import MarketRegime

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Rolling window length (steps) for per-agent performance tracking.
_PERF_WINDOW: int = 200

#: Softmax temperature for converting performance scores to weights.
_SOFTMAX_TEMP: float = 1.0

#: Minimum weight floor to keep all agents in the ensemble.
_MIN_WEIGHT: float = 0.05


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------


@dataclass
class AgentPerformanceRecord:
    """Rolling correctness tracker for a single agent in a given regime.

    Attributes:
        agent_name: Human-readable agent identifier.
        regime:     The :class:`~regime_detector.MarketRegime` this record tracks.
        window:     Deque of 1 (correct) / 0 (incorrect) outcomes.
    """

    agent_name: str
    regime: MarketRegime
    window: Deque[int] = field(default_factory=lambda: deque(maxlen=_PERF_WINDOW))

    def record(self, correct: bool) -> None:
        """Append a correctness result to the rolling window."""
        self.window.append(int(correct))

    @property
    def accuracy(self) -> float:
        """Rolling accuracy; defaults to 0.5 (random) when window is empty."""
        if not self.window:
            return 0.5
        return float(np.mean(self.window))


class PerformanceTracker:
    """Tracks per-(agent, regime) rolling accuracy.

    Args:
        agent_names: List of agent identifier strings.
        window_size: Rolling window length (default :data:`_PERF_WINDOW`).
    """

    def __init__(self, agent_names: List[str], window_size: int = _PERF_WINDOW) -> None:
        self._window_size = window_size
        self._records: Dict[Tuple[str, MarketRegime], AgentPerformanceRecord] = {}
        for name in agent_names:
            for regime in MarketRegime:
                self._records[(name, regime)] = AgentPerformanceRecord(
                    agent_name=name,
                    regime=regime,
                    window=deque(maxlen=window_size),
                )

    def record(self, agent_name: str, regime: MarketRegime, correct: bool) -> None:
        """Record one outcome for the given agent in the given regime.

        Args:
            agent_name: Must match a name supplied at construction.
            regime:     Current market regime.
            correct:    Whether the agent's action was directionally correct.
        """
        key = (agent_name, regime)
        if key not in self._records:
            log.warning("perf_tracker.unknown_agent", agent_name=agent_name)
            return
        self._records[key].record(correct)

    def accuracy(self, agent_name: str, regime: MarketRegime) -> float:
        """Return rolling accuracy for (agent, regime).

        Returns:
            Float in ``[0, 1]``; 0.5 if no data yet.
        """
        return self._records.get((agent_name, regime), AgentPerformanceRecord(agent_name, regime)).accuracy

    def weights(self, agent_names: List[str], regime: MarketRegime, temp: float = _SOFTMAX_TEMP) -> np.ndarray:
        """Compute softmax weights from per-agent accuracies in the given regime.

        Args:
            agent_names: Ordered list of agent names.
            regime:      Current market regime.
            temp:        Softmax temperature; lower = sharper.

        Returns:
            1-D numpy array of weights summing to 1, length ``len(agent_names)``.
        """
        accs = np.array([self.accuracy(n, regime) for n in agent_names], dtype=np.float64)
        # Softmax with temperature
        logits = accs / max(temp, 1e-9)
        logits -= logits.max()  # numerical stability
        exp = np.exp(logits)
        raw_weights = exp / exp.sum()
        # Apply minimum floor and renormalise
        floored = np.maximum(raw_weights, _MIN_WEIGHT)
        return floored / floored.sum()


# ---------------------------------------------------------------------------
# MetaEnsemble
# ---------------------------------------------------------------------------


class MetaEnsemble:
    """Combines DQN, D3QN, and DDPG-style agents with an online meta-learner.

    The meta-learner weights each agent's action vote by its rolling accuracy
    in the current detected regime, then takes a softmax-weighted vote.

    Args:
        agents:        Dict mapping agent name -> agent instance.  Each instance
                       must expose a ``select_actions(state)`` method that returns
                       a ``Tensor`` of shape ``(1,)`` with values in ``{0, 1, 2}``
                       (mapped internally to ``{-1, 0, +1}``).
        action_dim:    Action space size (default 3: sell/hold/buy).
        softmax_temp:  Temperature for meta-learner weight computation.
        perf_window:   Rolling window length for accuracy tracking.

    Example::

        action, weights = ensemble.act(state, regime=MarketRegime.BULL)
        ensemble.update_performance(actions, realised_return, regime)
    """

    # Map raw agent output index -> signed action
    _IDX_TO_ACTION: dict[int, int] = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        agents: Dict[str, Any],
        action_dim: int = 3,
        softmax_temp: float = _SOFTMAX_TEMP,
        perf_window: int = _PERF_WINDOW,
    ) -> None:
        self.agents = agents
        self.agent_names: List[str] = list(agents.keys())
        self.action_dim = action_dim
        self.softmax_temp = softmax_temp

        self.tracker = PerformanceTracker(
            agent_names=self.agent_names,
            window_size=perf_window,
        )

        # Most recent raw actions per agent (for deferred performance update)
        self._last_actions: Dict[str, int] = {}
        self._last_regime: MarketRegime = MarketRegime.SIDEWAYS

        log.info(
            "meta_ensemble.init",
            agents=self.agent_names,
            softmax_temp=softmax_temp,
            perf_window=perf_window,
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(
        self,
        state: Tensor,
        regime: MarketRegime = MarketRegime.SIDEWAYS,
    ) -> Tuple[int, Dict[str, float]]:
        """Compute the ensemble action for the given state and regime.

        Args:
            state:  Environment observation tensor, shape ``(state_dim,)`` or
                    ``(1, state_dim)``.
            regime: Current market regime from the HMM detector.

        Returns:
            A tuple ``(final_action, weight_dict)`` where *final_action* is in
            ``{-1, 0, +1}`` and *weight_dict* maps each agent name to its weight.
        """
        self._last_regime = regime
        raw_actions: Dict[str, int] = {}

        # Collect per-agent actions
        for name, agent in self.agents.items():
            try:
                with torch.no_grad():
                    out = agent.select_actions(state)
                    idx = int(out.squeeze().item())
                    action = self._IDX_TO_ACTION.get(idx, 0)
            except Exception as exc:
                log.warning("meta_ensemble.agent_error", agent=name, error=str(exc))
                action = 0
            raw_actions[name] = action

        self._last_actions = raw_actions

        # Compute meta-weights
        weights_arr = self.tracker.weights(self.agent_names, regime, self.softmax_temp)
        weight_dict = dict(zip(self.agent_names, weights_arr.tolist()))

        # Weighted vote
        vote = sum(weight_dict[n] * raw_actions[n] for n in self.agent_names)
        if vote > 1e-6:
            final_action = 1
        elif vote < -1e-6:
            final_action = -1
        else:
            final_action = 0

        log.info(
            "meta_ensemble.act",
            regime=regime.name,
            raw_actions=raw_actions,
            weights={k: round(v, 4) for k, v in weight_dict.items()},
            vote=round(float(vote), 4),
            final_action=final_action,
        )
        return final_action, weight_dict

    # ------------------------------------------------------------------
    # Performance feedback
    # ------------------------------------------------------------------

    def update_performance(
        self,
        realised_return: float,
        regime: Optional[MarketRegime] = None,
    ) -> None:
        """Update per-agent accuracy from the realised return on the last step.

        An agent's action is considered *correct* if::

            sign(action) == sign(realised_return)  and  abs(realised_return) > 0

        Hold (action == 0) is never penalised nor rewarded.

        Args:
            realised_return: Price change fraction over the held position period.
            regime:          Regime at the time the action was taken.  Defaults
                             to the regime recorded in the last :meth:`act` call.
        """
        used_regime = regime or self._last_regime

        for name, action in self._last_actions.items():
            if action == 0:
                continue  # hold: no feedback
            correct = (
                (action > 0 and realised_return > 0)
                or (action < 0 and realised_return < 0)
            )
            self.tracker.record(name, used_regime, correct)

        log.debug(
            "meta_ensemble.perf_update",
            regime=used_regime.name,
            realised_return=round(float(realised_return), 6),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def regime_accuracy_table(self) -> Dict[str, Dict[str, float]]:
        """Return per-(agent, regime) accuracy table for logging / dashboards.

        Returns:
            Nested dict: ``{agent_name: {regime_name: accuracy}}``.
        """
        table: Dict[str, Dict[str, float]] = {}
        for name in self.agent_names:
            table[name] = {
                regime.name: round(self.tracker.accuracy(name, regime), 4)
                for regime in MarketRegime
            }
        return table

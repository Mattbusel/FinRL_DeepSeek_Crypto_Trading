"""Meta-learning agent that learns WHEN to trade based on ensemble performance history.

The :class:`MetaAgent` wraps the existing :class:`MetaEnsemble` with a
higher-level controller that observes rolling ensemble accuracy over three
time horizons (1 h, 4 h, 24 h) and adjusts position sizing accordingly:

- **Uncertainty regime** (accuracy < threshold): position multiplier = 0.25
- **Confidence regime** (accuracy > threshold): position multiplier = 1.50
- **Normal regime** (between thresholds): position multiplier = 1.00

Regime transitions are recorded in a :class:`RegimeTransitionLog` for post-hoc
analysis and performance attribution.

Example::

    from meta_agent import MetaAgent, MetaState, MetaRegime
    from meta_ensemble import MetaEnsemble

    agent = MetaAgent(ensemble=ensemble, bars_per_hour=60)
    state, multiplier = agent.observe_and_decide(
        predicted_action=1,
        realised_return=0.002,
    )
    print(f"Regime={state.regime.name}  mult={multiplier:.2f}")

    report = agent.transition_log.as_dataframe()
    report.to_csv("regime_transitions.csv", index=False)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Rolling windows expressed in bars.  The caller supplies ``bars_per_hour``
#: at construction so the windows adapt to any data frequency.
_WINDOW_1H_BARS: int = 60    # default: 1 bar = 1 minute → 60 bars = 1 h
_WINDOW_4H_BARS: int = 240
_WINDOW_24H_BARS: int = 1440

#: Accuracy below this threshold triggers the uncertainty regime.
_UNCERTAINTY_THRESHOLD: float = 0.45

#: Accuracy above this threshold triggers the confidence regime.
_CONFIDENCE_THRESHOLD: float = 0.60

#: Position size multipliers for each regime.
_MULTIPLIER_UNCERTAINTY: float = 0.25
_MULTIPLIER_NORMAL: float = 1.00
_MULTIPLIER_CONFIDENCE: float = 1.50


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------


class MetaRegime(Enum):
    """Three-state meta-regime used to scale position sizes."""
    UNCERTAINTY = auto()   # low accuracy → reduce exposure
    NORMAL      = auto()   # moderate accuracy → baseline sizing
    CONFIDENCE  = auto()   # high accuracy → increase exposure


@dataclass
class MetaState:
    """Snapshot of the meta-agent's current perception of ensemble quality.

    Attributes:
        accuracy_1h:        Rolling accuracy over the past 1-hour window.
        accuracy_4h:        Rolling accuracy over the past 4-hour window.
        accuracy_24h:       Rolling accuracy over the past 24-hour window.
        regime:             The :class:`MetaRegime` derived from the above.
        position_multiplier: Scalar to apply to the base position size.
        timestamp:          Wall-clock time (seconds since epoch) at creation.
    """
    accuracy_1h:         float
    accuracy_4h:         float
    accuracy_24h:        float
    regime:              MetaRegime
    position_multiplier: float
    timestamp:           float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"MetaState(regime={self.regime.name}, "
            f"mult={self.position_multiplier:.2f}, "
            f"acc_1h={self.accuracy_1h:.3f}, "
            f"acc_4h={self.accuracy_4h:.3f}, "
            f"acc_24h={self.accuracy_24h:.3f})"
        )


@dataclass
class RegimeTransition:
    """A single regime-change event recorded for post-hoc analysis.

    Attributes:
        from_regime:  Previous :class:`MetaRegime`.
        to_regime:    New :class:`MetaRegime`.
        timestamp:    Wall-clock time of the transition (seconds since epoch).
        accuracy_1h:  1-hour accuracy at the moment of transition.
        accuracy_4h:  4-hour accuracy at the moment of transition.
        accuracy_24h: 24-hour accuracy at the moment of transition.
        bar_index:    Global bar counter value at the moment of transition.
    """
    from_regime:  MetaRegime
    to_regime:    MetaRegime
    timestamp:    float
    accuracy_1h:  float
    accuracy_4h:  float
    accuracy_24h: float
    bar_index:    int


class RegimeTransitionLog:
    """Ordered log of :class:`RegimeTransition` events.

    Provides convenience helpers for serialisation and analysis.

    Args:
        max_entries: Maximum number of transitions to store in memory.
                     Oldest entries are dropped when the limit is reached.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: Deque[RegimeTransition] = deque(maxlen=max_entries)

    def record(self, transition: RegimeTransition) -> None:
        """Append a new transition to the log."""
        self._entries.append(transition)
        log.info(
            "meta_agent.regime_transition",
            from_regime=transition.from_regime.name,
            to_regime=transition.to_regime.name,
            acc_1h=round(transition.accuracy_1h, 4),
            acc_4h=round(transition.accuracy_4h, 4),
            acc_24h=round(transition.accuracy_24h, 4),
            bar_index=transition.bar_index,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[RegimeTransition]:
        """Return a snapshot list of all recorded transitions (oldest first)."""
        return list(self._entries)

    def as_dataframe(self):
        """Return transitions as a ``pandas.DataFrame``.

        Requires ``pandas`` to be installed.  Raises :class:`ImportError` if
        pandas is not available.

        Returns:
            DataFrame with columns:
            ``from_regime``, ``to_regime``, ``timestamp``, ``accuracy_1h``,
            ``accuracy_4h``, ``accuracy_24h``, ``bar_index``.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for RegimeTransitionLog.as_dataframe(). "
                "Install it with: pip install pandas"
            )
        rows = [
            {
                "from_regime":  e.from_regime.name,
                "to_regime":    e.to_regime.name,
                "timestamp":    e.timestamp,
                "accuracy_1h":  e.accuracy_1h,
                "accuracy_4h":  e.accuracy_4h,
                "accuracy_24h": e.accuracy_24h,
                "bar_index":    e.bar_index,
            }
            for e in self._entries
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Return a summary dict with per-transition-type counts.

        Returns:
            Dict with keys ``total``, ``to_uncertainty``, ``to_normal``,
            ``to_confidence``.
        """
        total = len(self._entries)
        to_uncertainty = sum(
            1 for e in self._entries if e.to_regime is MetaRegime.UNCERTAINTY
        )
        to_confidence = sum(
            1 for e in self._entries if e.to_regime is MetaRegime.CONFIDENCE
        )
        return {
            "total": total,
            "to_uncertainty": to_uncertainty,
            "to_normal": total - to_uncertainty - to_confidence,
            "to_confidence": to_confidence,
        }


# ---------------------------------------------------------------------------
# Rolling accuracy tracker
# ---------------------------------------------------------------------------


class _RollingAccuracyTracker:
    """Maintains three deques tracking per-bar correctness across windows.

    Args:
        bars_1h:  Number of bars in the 1-hour window.
        bars_4h:  Number of bars in the 4-hour window.
        bars_24h: Number of bars in the 24-hour window.
    """

    def __init__(self, bars_1h: int, bars_4h: int, bars_24h: int) -> None:
        self._w1:  Deque[int] = deque(maxlen=bars_1h)
        self._w4:  Deque[int] = deque(maxlen=bars_4h)
        self._w24: Deque[int] = deque(maxlen=bars_24h)

    def update(self, correct: bool) -> None:
        """Record one correctness outcome across all windows.

        Args:
            correct: True if the ensemble prediction was directionally correct.
        """
        val = int(correct)
        self._w1.append(val)
        self._w4.append(val)
        self._w24.append(val)

    def _mean(self, dq: Deque[int]) -> float:
        """Return the mean of a deque, defaulting to 0.5 when empty."""
        if not dq:
            return 0.5
        return float(np.mean(dq))

    @property
    def accuracy_1h(self) -> float:
        """Rolling accuracy for the 1-hour window."""
        return self._mean(self._w1)

    @property
    def accuracy_4h(self) -> float:
        """Rolling accuracy for the 4-hour window."""
        return self._mean(self._w4)

    @property
    def accuracy_24h(self) -> float:
        """Rolling accuracy for the 24-hour window."""
        return self._mean(self._w24)

    def accuracies(self) -> Tuple[float, float, float]:
        """Return (accuracy_1h, accuracy_4h, accuracy_24h)."""
        return self.accuracy_1h, self.accuracy_4h, self.accuracy_24h


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------


class MetaAgent:
    """Meta-learning layer that controls WHEN and HOW MUCH to trade.

    Observes ensemble prediction accuracy over three rolling time horizons and
    transitions between three regimes:

    - **UNCERTAINTY**: accuracy < :data:`_UNCERTAINTY_THRESHOLD`
      → position multiplier = **0.25** (quarter size)
    - **NORMAL**: accuracy in [threshold_low, threshold_high]
      → position multiplier = **1.00** (baseline)
    - **CONFIDENCE**: accuracy > :data:`_CONFIDENCE_THRESHOLD`
      → position multiplier = **1.50** (50 % oversize)

    The regime is determined by a majority vote across all three windows; the
    24-hour window acts as a low-pass filter that dampens short-term noise.

    Args:
        bars_per_hour:          Number of data bars per hour (e.g. 60 for
                                1-minute bars, 4 for 15-minute bars).
        uncertainty_threshold:  Accuracy below which UNCERTAINTY is signalled.
        confidence_threshold:   Accuracy above which CONFIDENCE is signalled.
        multiplier_uncertainty: Position multiplier in the UNCERTAINTY regime.
        multiplier_normal:      Position multiplier in the NORMAL regime.
        multiplier_confidence:  Position multiplier in the CONFIDENCE regime.
        log_transitions:        If True, regime changes are logged at INFO level.

    Example::

        from meta_agent import MetaAgent, MetaRegime

        agent = MetaAgent(bars_per_hour=60)

        for bar in live_feed:
            predicted_action = ensemble.act(bar.state)[0]
            state, multiplier = agent.observe_and_decide(
                predicted_action=predicted_action,
                realised_return=bar.realised_return,
            )
            final_size = base_size * multiplier
    """

    def __init__(
        self,
        bars_per_hour: int = _WINDOW_1H_BARS,
        uncertainty_threshold: float = _UNCERTAINTY_THRESHOLD,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        multiplier_uncertainty: float = _MULTIPLIER_UNCERTAINTY,
        multiplier_normal: float = _MULTIPLIER_NORMAL,
        multiplier_confidence: float = _MULTIPLIER_CONFIDENCE,
        log_transitions: bool = True,
    ) -> None:
        if bars_per_hour <= 0:
            raise ValueError(f"bars_per_hour must be positive, got {bars_per_hour}")
        if not (0.0 < uncertainty_threshold < confidence_threshold < 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 < uncertainty < confidence < 1, "
                f"got {uncertainty_threshold} and {confidence_threshold}"
            )

        self._unc_thresh   = uncertainty_threshold
        self._conf_thresh  = confidence_threshold
        self._mult_unc     = multiplier_uncertainty
        self._mult_norm    = multiplier_normal
        self._mult_conf    = multiplier_confidence
        self._log_trans    = log_transitions

        bars_1h  = bars_per_hour
        bars_4h  = bars_per_hour * 4
        bars_24h = bars_per_hour * 24

        self._tracker = _RollingAccuracyTracker(
            bars_1h=bars_1h,
            bars_4h=bars_4h,
            bars_24h=bars_24h,
        )

        self._current_regime: MetaRegime = MetaRegime.NORMAL
        self._bar_index: int = 0
        self.transition_log: RegimeTransitionLog = RegimeTransitionLog()

        log.info(
            "meta_agent.init",
            bars_per_hour=bars_per_hour,
            windows=(bars_1h, bars_4h, bars_24h),
            thresholds=(uncertainty_threshold, confidence_threshold),
            multipliers=(multiplier_uncertainty, multiplier_normal, multiplier_confidence),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe_and_decide(
        self,
        predicted_action: int,
        realised_return: float,
    ) -> Tuple[MetaState, float]:
        """Update accuracy trackers and return the current :class:`MetaState`.

        Call this once per bar **after** the bar's realised return is known.

        Args:
            predicted_action: The ensemble's action for this bar
                              (``+1`` = buy, ``-1`` = sell, ``0`` = hold).
            realised_return:  The price return realised over the bar.
                              Used to score the predicted_action's correctness.

        Returns:
            A tuple ``(MetaState, position_multiplier)`` where
            *position_multiplier* is the scalar to apply to base position size.
        """
        # Score correctness: hold actions (0) are not penalised.
        if predicted_action != 0:
            correct = (
                (predicted_action > 0 and realised_return > 0) or
                (predicted_action < 0 and realised_return < 0)
            )
            self._tracker.update(correct)
        else:
            # For hold actions feed nothing; windows naturally age out.
            pass

        self._bar_index += 1
        acc_1h, acc_4h, acc_24h = self._tracker.accuracies()
        new_regime = self._classify_regime(acc_1h, acc_4h, acc_24h)

        # Record transition if regime changed.
        if new_regime is not self._current_regime:
            transition = RegimeTransition(
                from_regime=self._current_regime,
                to_regime=new_regime,
                timestamp=time.time(),
                accuracy_1h=acc_1h,
                accuracy_4h=acc_4h,
                accuracy_24h=acc_24h,
                bar_index=self._bar_index,
            )
            self.transition_log.record(transition)
            self._current_regime = new_regime

        multiplier = self._regime_to_multiplier(self._current_regime)
        state = MetaState(
            accuracy_1h=acc_1h,
            accuracy_4h=acc_4h,
            accuracy_24h=acc_24h,
            regime=self._current_regime,
            position_multiplier=multiplier,
        )

        log.debug(
            "meta_agent.step",
            bar=self._bar_index,
            action=predicted_action,
            ret=round(realised_return, 6),
            acc_1h=round(acc_1h, 4),
            regime=self._current_regime.name,
            mult=multiplier,
        )
        return state, multiplier

    def current_state(self) -> MetaState:
        """Return the current :class:`MetaState` without consuming a bar.

        Useful for dashboard queries between bars.

        Returns:
            MetaState snapshot based on the latest accuracy windows.
        """
        acc_1h, acc_4h, acc_24h = self._tracker.accuracies()
        regime = self._classify_regime(acc_1h, acc_4h, acc_24h)
        multiplier = self._regime_to_multiplier(regime)
        return MetaState(
            accuracy_1h=acc_1h,
            accuracy_4h=acc_4h,
            accuracy_24h=acc_24h,
            regime=regime,
            position_multiplier=multiplier,
        )

    def reset(self) -> None:
        """Reset all accuracy windows and return to NORMAL regime.

        Useful at session boundaries.  The transition log is preserved.
        """
        bars_1h  = self._tracker._w1.maxlen or _WINDOW_1H_BARS
        bars_4h  = self._tracker._w4.maxlen or _WINDOW_4H_BARS
        bars_24h = self._tracker._w24.maxlen or _WINDOW_24H_BARS
        self._tracker = _RollingAccuracyTracker(
            bars_1h=bars_1h,
            bars_4h=bars_4h,
            bars_24h=bars_24h,
        )
        self._current_regime = MetaRegime.NORMAL
        self._bar_index = 0
        log.info("meta_agent.reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        acc_1h: float,
        acc_4h: float,
        acc_24h: float,
    ) -> MetaRegime:
        """Classify regime via majority vote across the three windows.

        Each window votes independently:
        - accuracy < _unc_thresh  → UNCERTAINTY vote
        - accuracy > _conf_thresh → CONFIDENCE vote
        - otherwise               → NORMAL vote

        The regime with the most votes wins.  Ties are broken in the order
        NORMAL > UNCERTAINTY > CONFIDENCE (conservative tie-breaking).

        Args:
            acc_1h:  1-hour rolling accuracy.
            acc_4h:  4-hour rolling accuracy.
            acc_24h: 24-hour rolling accuracy.

        Returns:
            The majority :class:`MetaRegime`.
        """
        votes: dict[MetaRegime, int] = {
            MetaRegime.UNCERTAINTY: 0,
            MetaRegime.NORMAL:      0,
            MetaRegime.CONFIDENCE:  0,
        }

        for acc in (acc_1h, acc_4h, acc_24h):
            if acc < self._unc_thresh:
                votes[MetaRegime.UNCERTAINTY] += 1
            elif acc > self._conf_thresh:
                votes[MetaRegime.CONFIDENCE] += 1
            else:
                votes[MetaRegime.NORMAL] += 1

        # Conservative tie-breaking order: NORMAL wins ties.
        if votes[MetaRegime.NORMAL] >= votes[MetaRegime.UNCERTAINTY] and \
                votes[MetaRegime.NORMAL] >= votes[MetaRegime.CONFIDENCE]:
            return MetaRegime.NORMAL
        if votes[MetaRegime.UNCERTAINTY] >= votes[MetaRegime.CONFIDENCE]:
            return MetaRegime.UNCERTAINTY
        return MetaRegime.CONFIDENCE

    def _regime_to_multiplier(self, regime: MetaRegime) -> float:
        """Map a regime to its position size multiplier.

        Args:
            regime: The current :class:`MetaRegime`.

        Returns:
            Float position size multiplier.
        """
        _map = {
            MetaRegime.UNCERTAINTY: self._mult_unc,
            MetaRegime.NORMAL:      self._mult_norm,
            MetaRegime.CONFIDENCE:  self._mult_conf,
        }
        return _map[regime]

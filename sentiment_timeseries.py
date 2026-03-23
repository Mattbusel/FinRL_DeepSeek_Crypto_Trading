"""Sentiment time-series analysis for DeepSeek-derived signals.

Tracks DeepSeek sentiment scores over time, builds a 7-day rolling sentiment
momentum indicator, and detects sentiment reversals as actionable trading signals.

Pipeline
--------
1. :class:`SentimentRecord` — timestamped (sentiment_score, confidence) pair.
2. :class:`SentimentTimeSeries` — in-memory store of records with rolling
   statistics and momentum computation.
3. :class:`SentimentReversalDetector` — emits :class:`ReversalSignal` events
   when the 7-day rolling momentum crosses zero or exceeds a threshold.

Sentiment scores are assumed to follow the DeepSeek convention: 1 (very
negative) … 5 (very positive).  Internally they are normalised to [-1, +1].

Example::

    from sentiment_timeseries import SentimentTimeSeries, SentimentReversalDetector
    import datetime

    ts = SentimentTimeSeries(window_days=7)
    ts.add(score=4.2, confidence=0.9, timestamp=datetime.datetime.utcnow())
    ...
    detector = SentimentReversalDetector(ts)
    signals = detector.detect()
    for s in signals:
        print(s)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence

import numpy as np

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: DeepSeek score range for normalisation.
_SCORE_MIN: float = 1.0
_SCORE_MAX: float = 5.0

#: Default rolling window in calendar days.
_DEFAULT_WINDOW_DAYS: int = 7

#: Momentum threshold that triggers a reversal signal (absolute value).
_REVERSAL_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(order=True)
class SentimentRecord:
    """A single timestamped sentiment observation.

    Attributes:
        timestamp:   UTC datetime of the observation.
        raw_score:   Raw DeepSeek sentiment score in ``[1, 5]``.
        confidence:  Model confidence in ``[0, 1]``.
        normalised:  Score normalised to ``[-1, +1]`` (derived).
    """

    timestamp: datetime.datetime
    raw_score: float
    confidence: float
    normalised: float = field(init=False)

    def __post_init__(self) -> None:
        self.normalised = (self.raw_score - _SCORE_MIN) / (_SCORE_MAX - _SCORE_MIN) * 2.0 - 1.0


class ReversalDirection(Enum):
    """Direction of a detected sentiment reversal."""

    BULLISH_REVERSAL = auto()   # momentum turns positive (bearish -> bullish)
    BEARISH_REVERSAL = auto()   # momentum turns negative (bullish -> bearish)
    MOMENTUM_SPIKE_UP = auto()  # extreme positive momentum surge
    MOMENTUM_SPIKE_DOWN = auto()  # extreme negative momentum plunge


@dataclass
class ReversalSignal:
    """A trading signal derived from a sentiment reversal event.

    Attributes:
        timestamp:      When the reversal was detected.
        direction:      :class:`ReversalDirection` enum value.
        momentum:       Current 7-day rolling sentiment momentum.
        prev_momentum:  Momentum at the previous observation.
        confidence:     Average confidence weight over the rolling window.
        suggested_action: ``1`` (buy), ``-1`` (sell), or ``0`` (monitor).
    """

    timestamp: datetime.datetime
    direction: ReversalDirection
    momentum: float
    prev_momentum: float
    confidence: float
    suggested_action: int  # 1, -1, 0


# ---------------------------------------------------------------------------
# SentimentTimeSeries
# ---------------------------------------------------------------------------


class SentimentTimeSeries:
    """In-memory time-ordered store of sentiment records with rolling analytics.

    Args:
        window_days:  Number of calendar days in the rolling window (default 7).
        max_records:  Maximum records retained in memory (oldest pruned first).
                      Default is 10 000.

    Example::

        ts = SentimentTimeSeries(window_days=7)
        ts.add(score=3.8, confidence=0.85)
        mom = ts.rolling_momentum()
    """

    def __init__(
        self,
        window_days: int = _DEFAULT_WINDOW_DAYS,
        max_records: int = 10_000,
    ) -> None:
        self.window_days = window_days
        self.max_records = max_records
        self._records: List[SentimentRecord] = []

        log.info(
            "sentiment_timeseries.init",
            window_days=window_days,
            max_records=max_records,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        score: float,
        confidence: float = 1.0,
        timestamp: Optional[datetime.datetime] = None,
    ) -> SentimentRecord:
        """Append a new sentiment observation.

        Args:
            score:      Raw DeepSeek sentiment score in ``[1, 5]``.
            confidence: Model confidence in ``[0, 1]`` (default 1.0).
            timestamp:  UTC datetime; defaults to ``datetime.datetime.utcnow()``.

        Returns:
            The created :class:`SentimentRecord`.

        Raises:
            ValueError: If ``score`` is outside ``[1, 5]`` or ``confidence``
                outside ``[0, 1]``.
        """
        if not (_SCORE_MIN <= score <= _SCORE_MAX):
            raise ValueError(f"score must be in [{_SCORE_MIN}, {_SCORE_MAX}]; got {score}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]; got {confidence}")

        ts = timestamp or datetime.datetime.utcnow()
        record = SentimentRecord(timestamp=ts, raw_score=score, confidence=confidence)

        self._records.append(record)
        self._records.sort(key=lambda r: r.timestamp)

        # Prune oldest if over limit
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        log.debug(
            "sentiment_ts.add",
            score=score,
            normalised=round(record.normalised, 4),
            confidence=confidence,
            timestamp=ts.isoformat(),
        )
        return record

    # ------------------------------------------------------------------
    # Windowed access
    # ------------------------------------------------------------------

    def window(self, as_of: Optional[datetime.datetime] = None) -> List[SentimentRecord]:
        """Return records within the last ``window_days`` days.

        Args:
            as_of: Reference datetime (default: utcnow).

        Returns:
            Chronologically ordered list of :class:`SentimentRecord` in window.
        """
        ref = as_of or datetime.datetime.utcnow()
        cutoff = ref - datetime.timedelta(days=self.window_days)
        return [r for r in self._records if r.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def rolling_mean(self, as_of: Optional[datetime.datetime] = None) -> Optional[float]:
        """Confidence-weighted mean normalised sentiment in the rolling window.

        Returns:
            Float in ``[-1, +1]``, or ``None`` if the window is empty.
        """
        recs = self.window(as_of)
        if not recs:
            return None
        scores = np.array([r.normalised for r in recs])
        weights = np.array([r.confidence for r in recs])
        total_w = weights.sum()
        if total_w < 1e-9:
            return float(scores.mean())
        return float(np.average(scores, weights=weights))

    def rolling_std(self, as_of: Optional[datetime.datetime] = None) -> Optional[float]:
        """Standard deviation of normalised sentiment in the rolling window.

        Returns:
            Float >= 0, or ``None`` if the window has < 2 records.
        """
        recs = self.window(as_of)
        if len(recs) < 2:
            return None
        return float(np.std([r.normalised for r in recs]))

    def rolling_momentum(
        self,
        half_window: Optional[int] = None,
        as_of: Optional[datetime.datetime] = None,
    ) -> Optional[float]:
        """7-day rolling sentiment momentum: mean(recent half) - mean(older half).

        Splits the rolling window at the midpoint and subtracts the mean of the
        older half from the mean of the newer half.  Positive values indicate
        improving sentiment; negative values indicate deteriorating sentiment.

        Args:
            half_window: Override the auto-split midpoint (in record count).
            as_of:       Reference datetime.

        Returns:
            Momentum float in ``[-2, +2]``, or ``None`` if < 4 records in window.
        """
        recs = self.window(as_of)
        if len(recs) < 4:
            return None

        mid = half_window if half_window is not None else len(recs) // 2
        older = recs[:mid]
        newer = recs[mid:]

        def wmean(rs: List[SentimentRecord]) -> float:
            scores = np.array([r.normalised for r in rs])
            weights = np.array([r.confidence for r in rs])
            tw = weights.sum()
            return float(np.average(scores, weights=weights)) if tw > 1e-9 else float(scores.mean())

        momentum = wmean(newer) - wmean(older)
        log.debug(
            "sentiment_ts.momentum",
            momentum=round(momentum, 6),
            n_older=len(older),
            n_newer=len(newer),
        )
        return momentum

    def all_scores_normalised(self) -> List[float]:
        """Return all stored normalised scores in chronological order."""
        return [r.normalised for r in self._records]

    def average_confidence(self, as_of: Optional[datetime.datetime] = None) -> float:
        """Mean confidence of records in the rolling window."""
        recs = self.window(as_of)
        if not recs:
            return 0.0
        return float(np.mean([r.confidence for r in recs]))

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"SentimentTimeSeries(records={len(self._records)}, "
            f"window_days={self.window_days})"
        )


# ---------------------------------------------------------------------------
# SentimentReversalDetector
# ---------------------------------------------------------------------------


class SentimentReversalDetector:
    """Detects sentiment reversals in a :class:`SentimentTimeSeries`.

    Emits :class:`ReversalSignal` when:

    * Momentum crosses zero (sign change) between consecutive detections.
    * Absolute momentum exceeds ``spike_threshold`` (extreme move).

    Args:
        timeseries:       The :class:`SentimentTimeSeries` to monitor.
        reversal_threshold: Minimum momentum magnitude for zero-cross to count.
        spike_threshold:    Momentum magnitude at which a spike signal is emitted
                            regardless of sign change.

    Example::

        detector = SentimentReversalDetector(ts, reversal_threshold=0.10)
        for signal in detector.detect():
            print(signal.direction, signal.suggested_action)
    """

    def __init__(
        self,
        timeseries: SentimentTimeSeries,
        reversal_threshold: float = _REVERSAL_THRESHOLD,
        spike_threshold: float = 0.40,
    ) -> None:
        self.ts = timeseries
        self.reversal_threshold = reversal_threshold
        self.spike_threshold = spike_threshold
        self._prev_momentum: Optional[float] = None

        log.info(
            "reversal_detector.init",
            reversal_threshold=reversal_threshold,
            spike_threshold=spike_threshold,
        )

    def detect(
        self, as_of: Optional[datetime.datetime] = None
    ) -> List[ReversalSignal]:
        """Run detection against the current state of the time series.

        Args:
            as_of: Reference datetime for rolling window computation.

        Returns:
            List of :class:`ReversalSignal` objects (may be empty).
        """
        signals: List[ReversalSignal] = []
        now = as_of or datetime.datetime.utcnow()
        momentum = self.ts.rolling_momentum(as_of=now)

        if momentum is None:
            log.debug("reversal_detector.no_data")
            return signals

        prev = self._prev_momentum
        avg_conf = self.ts.average_confidence(as_of=now)

        # --- Spike detection (evaluated first, may overlap with reversal) ---
        if abs(momentum) >= self.spike_threshold:
            direction = (
                ReversalDirection.MOMENTUM_SPIKE_UP
                if momentum > 0
                else ReversalDirection.MOMENTUM_SPIKE_DOWN
            )
            signals.append(
                ReversalSignal(
                    timestamp=now,
                    direction=direction,
                    momentum=momentum,
                    prev_momentum=prev if prev is not None else 0.0,
                    confidence=avg_conf,
                    suggested_action=1 if momentum > 0 else -1,
                )
            )

        # --- Zero-cross reversal detection ---
        if prev is not None and abs(momentum) >= self.reversal_threshold:
            if prev <= 0.0 < momentum:
                signals.append(
                    ReversalSignal(
                        timestamp=now,
                        direction=ReversalDirection.BULLISH_REVERSAL,
                        momentum=momentum,
                        prev_momentum=prev,
                        confidence=avg_conf,
                        suggested_action=1,
                    )
                )
            elif prev >= 0.0 > momentum:
                signals.append(
                    ReversalSignal(
                        timestamp=now,
                        direction=ReversalDirection.BEARISH_REVERSAL,
                        momentum=momentum,
                        prev_momentum=prev,
                        confidence=avg_conf,
                        suggested_action=-1,
                    )
                )

        self._prev_momentum = momentum

        if signals:
            log.info(
                "reversal_detector.signals",
                n_signals=len(signals),
                directions=[s.direction.name for s in signals],
                momentum=round(momentum, 6),
            )

        return signals

    def reset(self) -> None:
        """Clear internal state (e.g., between backtest folds)."""
        self._prev_momentum = None
        log.debug("reversal_detector.reset")

"""Regime-aware position sizing for the LARSA trading system.

Integrates HMM-based market regime detection with Kelly criterion to produce
dynamically adjusted position sizes.  The sizing table is:

- BULL:             100% of base size
- SIDEWAYS:          60% of base size
- BEAR:              25% of base size
- HIGH_VOLATILITY:   10% of base size (protective)

Kelly criterion is applied *after* the regime multiplier so that the final
size is bounded both by volatility-regime context and bankroll-optimal theory.

Example::

    from regime_sizing import RegimeSizer
    from regime_detector import HMMRegimeDetector

    detector = HMMRegimeDetector().fit(train_prices)
    sizer = RegimeSizer(detector=detector, base_size=0.01, bankroll=100_000.0)
    size = sizer.compute_size(recent_prices, win_rate=0.55, avg_win=0.03, avg_loss=0.015)
    print(f"Position size: {size:.6f} BTC")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from exceptions import DataFetchError, ModelError
from logger import get_logger
from regime_detector import HMMRegimeDetector, MarketRegime

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regime multipliers
# ---------------------------------------------------------------------------

#: Fraction of base size to deploy in each regime.
REGIME_SIZE_MULTIPLIERS: dict[MarketRegime, float] = {
    MarketRegime.BULL: 1.00,
    MarketRegime.SIDEWAYS: 0.60,
    MarketRegime.BEAR: 0.25,
    MarketRegime.HIGH_VOLATILITY: 0.10,
}


# ---------------------------------------------------------------------------
# Kelly criterion helper
# ---------------------------------------------------------------------------


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction_cap: float = 0.25,
) -> float:
    """Compute the fractional Kelly position size.

    Uses the standard Kelly formula:
        f* = (b*p - q) / b
    where ``b = avg_win / avg_loss``, ``p = win_rate``, ``q = 1 - win_rate``.

    The result is capped at ``kelly_fraction_cap`` (default 0.25, i.e., quarter
    Kelly) to manage estimation error in ``win_rate`` / ``avg_win`` / ``avg_loss``.

    Args:
        win_rate:          Estimated probability of a winning trade (0 < p < 1).
        avg_win:           Average gain on winning trades (positive fraction).
        avg_loss:          Average loss on losing trades (positive fraction).
        kelly_fraction_cap: Upper bound on the returned Kelly fraction.

    Returns:
        Clipped Kelly fraction in ``[0.0, kelly_fraction_cap]``.

    Raises:
        ValueError: If ``avg_loss`` is zero or negative, or ``win_rate`` is
            outside (0, 1).
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError(f"win_rate must be in (0, 1); got {win_rate}")
    if avg_loss <= 0.0:
        raise ValueError(f"avg_loss must be positive; got {avg_loss}")
    if avg_win <= 0.0:
        raise ValueError(f"avg_win must be positive; got {avg_win}")

    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_star = (b * win_rate - q) / b

    clipped = float(np.clip(f_star, 0.0, kelly_fraction_cap))
    log.debug(
        "kelly_fraction.computed",
        raw_kelly=round(f_star, 6),
        clipped_kelly=round(clipped, 6),
        kelly_fraction_cap=kelly_fraction_cap,
    )
    return clipped


# ---------------------------------------------------------------------------
# RegimeSizer
# ---------------------------------------------------------------------------


@dataclass
class SizingDecision:
    """Output of a single :meth:`RegimeSizer.compute_size` call.

    Attributes:
        regime:           Detected market regime.
        regime_multiplier: Fraction of base size for this regime.
        kelly_f:          Kelly fraction computed from provided statistics.
        raw_size:         ``base_size * regime_multiplier``.
        kelly_size:       ``bankroll * kelly_f * regime_multiplier``.
        final_size:       Minimum of ``raw_size`` and ``kelly_size`` (most
                          conservative estimate).
    """

    regime: MarketRegime
    regime_multiplier: float
    kelly_f: float
    raw_size: float
    kelly_size: float
    final_size: float


class RegimeSizer:
    """Combines HMM regime detection with Kelly criterion for position sizing.

    Args:
        detector:           A *fitted* :class:`~regime_detector.HMMRegimeDetector`.
        base_size:          Nominal position size in asset units (e.g. BTC).
                            Used as the fallback when Kelly is unavailable.
        bankroll:           Current portfolio value in quote currency (e.g. USD).
                            Required for Kelly sizing.
        kelly_fraction_cap: Upper bound on the raw Kelly fraction (default 0.25).
        regime_multipliers: Optional override for :data:`REGIME_SIZE_MULTIPLIERS`.

    Example::

        sizer = RegimeSizer(detector=fitted_detector, base_size=0.005, bankroll=50_000)
        decision = sizer.compute_size(
            recent_prices=last_200_closes,
            win_rate=0.57,
            avg_win=0.025,
            avg_loss=0.012,
        )
        print(decision.final_size)
    """

    def __init__(
        self,
        detector: HMMRegimeDetector,
        base_size: float = 0.001,
        bankroll: float = 10_000.0,
        kelly_fraction_cap: float = 0.25,
        regime_multipliers: Optional[dict[MarketRegime, float]] = None,
    ) -> None:
        self.detector = detector
        self.base_size = base_size
        self.bankroll = bankroll
        self.kelly_fraction_cap = kelly_fraction_cap
        self.multipliers = regime_multipliers or REGIME_SIZE_MULTIPLIERS

        log.info(
            "regime_sizer.init",
            base_size=base_size,
            bankroll=bankroll,
            kelly_fraction_cap=kelly_fraction_cap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_size(
        self,
        recent_prices: npt.ArrayLike,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> SizingDecision:
        """Compute a regime-adjusted, Kelly-optimal position size.

        Steps:
        1. Detect current regime from ``recent_prices``.
        2. Look up the regime multiplier from :attr:`multipliers`.
        3. Compute Kelly fraction from ``win_rate``, ``avg_win``, ``avg_loss``.
        4. ``raw_size   = base_size * multiplier``
        5. ``kelly_size = bankroll * kelly_f * multiplier``
        6. ``final_size = min(raw_size, kelly_size)``

        Args:
            recent_prices: Recent close-price window (>= 22 elements).
            win_rate:      Historical win probability for this strategy.
            avg_win:       Average fractional gain on winning trades.
            avg_loss:      Average fractional loss on losing trades.

        Returns:
            A :class:`SizingDecision` with full breakdown.

        Raises:
            ModelError:     If the detector has not been fitted.
            DataFetchError: If ``recent_prices`` is too short.
            ValueError:     If Kelly parameters are invalid.
        """
        # --- 1. Regime detection ---
        try:
            regime = self.detector.predict(recent_prices)
        except (ModelError, DataFetchError) as exc:
            log.warning(
                "regime_sizer.predict_fallback",
                error=str(exc),
                fallback=MarketRegime.SIDEWAYS.name,
            )
            regime = MarketRegime.SIDEWAYS

        # --- 2. Regime multiplier ---
        multiplier = self.multipliers.get(regime, 0.60)

        # --- 3. Kelly fraction ---
        kelly_f = kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            kelly_fraction_cap=self.kelly_fraction_cap,
        )

        # --- 4-6. Size computation ---
        raw_size = self.base_size * multiplier
        kelly_size = self.bankroll * kelly_f * multiplier
        final_size = min(raw_size, kelly_size)

        decision = SizingDecision(
            regime=regime,
            regime_multiplier=multiplier,
            kelly_f=kelly_f,
            raw_size=raw_size,
            kelly_size=kelly_size,
            final_size=final_size,
        )

        log.info(
            "regime_sizer.decision",
            regime=regime.name,
            multiplier=multiplier,
            kelly_f=round(kelly_f, 6),
            raw_size=round(raw_size, 8),
            kelly_size=round(kelly_size, 8),
            final_size=round(final_size, 8),
        )
        return decision

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update the bankroll used for Kelly sizing.

        Args:
            new_bankroll: Current total portfolio value in quote currency.
        """
        log.info(
            "regime_sizer.bankroll_update",
            old=self.bankroll,
            new=new_bankroll,
        )
        self.bankroll = new_bankroll

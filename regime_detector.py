"""Hidden Markov Model-based market regime detection for the LARSA system.

Identifies one of four market regimes (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)
from a price series using a Gaussian HMM.  A :class:`RegimeAwareEnsemble`
wrapper adjusts per-agent voting weights based on the current regime to make
the ensemble more adaptive.

Dependencies::

    pip install hmmlearn

Example::

    import numpy as np
    from regime_detector import HMMRegimeDetector, MarketRegime

    detector = HMMRegimeDetector()
    detector.fit(price_series)
    regime = detector.predict(price_series[-60:])
    print(regime)  # MarketRegime.BULL
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np
import numpy.typing as npt

from exceptions import DataFetchError, ModelError
from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    """Four discrete market regimes distinguished by the HMM.

    Attributes:
        BULL: Trending upward with moderate volatility.
        BEAR: Trending downward with elevated volatility.
        SIDEWAYS: Low directional movement, low volatility.
        HIGH_VOLATILITY: Extreme volatility regardless of direction.
    """

    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()
    HIGH_VOLATILITY = auto()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _compute_features(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute a 4-feature matrix from a 1-D price array.

    Features:
        - ``returns``:       Log returns (price[t] / price[t-1]).
        - ``volatility``:    Rolling 20-step standard deviation of returns.
        - ``volume_ratio``:  Ratio of current abs-return to its 20-step mean
                             (proxy for relative volume/activity).
        - ``spread``:        Difference between max and min over a 5-step window
                             (proxy for intra-bar range / bid-ask spread).

    Args:
        prices: 1-D array of close prices with at least 21 elements.

    Returns:
        2-D feature matrix of shape ``(len(prices) - 20, 4)``.

    Raises:
        DataFetchError: If *prices* has fewer than 21 elements.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < 21:
        raise DataFetchError(
            f"Price series must be a 1-D array with >= 21 elements; got shape {prices.shape}."
        )

    returns = np.diff(np.log(np.clip(prices, 1e-10, None)))  # length N-1
    n = len(returns)

    # Rolling 20-step volatility
    vol = np.array(
        [returns[max(0, i - 20) : i + 1].std() for i in range(n)],
        dtype=np.float64,
    )

    # Volume ratio: abs_return / rolling 20-step mean abs_return
    abs_ret = np.abs(returns)
    rolling_mean_abs = np.array(
        [abs_ret[max(0, i - 20) : i + 1].mean() for i in range(n)],
        dtype=np.float64,
    )
    volume_ratio = np.where(rolling_mean_abs > 1e-12, abs_ret / rolling_mean_abs, 1.0)

    # Rolling 5-step spread (max - min); np.ptp removed in NumPy 2.x
    spread = np.array(
        [np.max(prices[max(0, i - 4) : i + 2]) - np.min(prices[max(0, i - 4) : i + 2])
         for i in range(n)],
        dtype=np.float64,
    )

    features = np.column_stack([returns, vol, volume_ratio, spread])  # (N-1, 4)
    # Trim to ensure alignment (drop first 20 rows to match rolling warm-up)
    return features[20:]


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """Gaussian HMM with 4 hidden states mapped to :class:`MarketRegime` values.

    Args:
        n_components: Number of HMM states.  Must be 4 to map to the four
            :class:`MarketRegime` values.  Advanced users may override.
        covariance_type: Covariance structure for GaussianHMM (default ``"full"``).
        n_iter: Maximum EM iterations (default 200).
        random_state: Seed for reproducibility (default 42).

    Raises:
        ConfigError: If ``hmmlearn`` is not installed.
    """

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: Any = None
        self._state_to_regime: dict[int, MarketRegime] = {}

        log.info(
            "regime_detector.init",
            n_components=n_components,
            covariance_type=covariance_type,
        )

    def _build_model(self) -> Any:
        """Instantiate a GaussianHMM from hmmlearn.

        Returns:
            Unfitted :class:`hmmlearn.hmm.GaussianHMM` instance.

        Raises:
            ModelError: If ``hmmlearn`` is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import]
        except ImportError as exc:
            raise ModelError(
                "hmmlearn is required for regime detection.  "
                "Run: pip install hmmlearn",
                cause=exc,
            ) from exc

        return GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, price_series: npt.ArrayLike) -> "HMMRegimeDetector":
        """Train the HMM on a historical price series.

        After fitting, hidden states are labelled by matching their mean
        return and volatility against regime definitions.

        Args:
            price_series: 1-D array-like of close prices (at least 100
                recommended for reliable convergence).

        Returns:
            ``self`` (fluent interface).

        Raises:
            DataFetchError: If the price series is too short.
            ModelError: If ``hmmlearn`` is not installed or fitting fails.
        """
        prices = np.asarray(price_series, dtype=np.float64)
        features = _compute_features(prices)

        self._model = self._build_model()
        try:
            self._model.fit(features)
        except Exception as exc:
            raise ModelError("HMM fitting failed.", cause=exc) from exc

        self._state_to_regime = self._label_states(features)

        log.info(
            "regime_detector.fit.done",
            n_samples=len(features),
            state_mapping={k: v.name for k, v in self._state_to_regime.items()},
        )
        return self

    def _label_states(self, features: npt.NDArray[np.float64]) -> dict[int, MarketRegime]:
        """Map HMM state indices to :class:`MarketRegime` labels.

        Heuristic assignment based on mean return and volatility of each state:

        - Highest mean return  -> BULL
        - Lowest mean return   -> BEAR
        - Highest volatility   -> HIGH_VOLATILITY  (overrides return-based label)
        - Remainder            -> SIDEWAYS

        Args:
            features: Feature matrix used to train the HMM.

        Returns:
            Dictionary mapping HMM state integer to :class:`MarketRegime`.
        """
        # Predict state sequence on training data for characterisation
        hidden_states = self._model.predict(features)

        mean_return: list[tuple[int, float]] = []
        mean_vol: list[tuple[int, float]] = []

        for state in range(self.n_components):
            mask = hidden_states == state
            if mask.any():
                mean_return.append((state, float(features[mask, 0].mean())))
                mean_vol.append((state, float(features[mask, 1].mean())))
            else:
                mean_return.append((state, 0.0))
                mean_vol.append((state, 0.0))

        sorted_by_return = sorted(mean_return, key=lambda x: x[1])
        sorted_by_vol = sorted(mean_vol, key=lambda x: x[1])

        mapping: dict[int, MarketRegime] = {}
        # Highest volatility state => HIGH_VOLATILITY
        high_vol_state = sorted_by_vol[-1][0]
        mapping[high_vol_state] = MarketRegime.HIGH_VOLATILITY

        # Remaining: assign by return rank
        remaining = [s for s, _ in sorted_by_return if s not in mapping]
        if len(remaining) >= 1:
            mapping[remaining[0]] = MarketRegime.BEAR
        if len(remaining) >= 2:
            mapping[remaining[-1]] = MarketRegime.BULL
        for s in remaining[1:-1]:
            if s not in mapping:
                mapping[s] = MarketRegime.SIDEWAYS

        # Ensure all states are covered
        for s in range(self.n_components):
            if s not in mapping:
                mapping[s] = MarketRegime.SIDEWAYS

        return mapping

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, recent_window: npt.ArrayLike) -> MarketRegime:
        """Predict the current market regime from a recent price window.

        Args:
            recent_window: 1-D array-like of recent close prices.  Should
                contain at least 22 elements to allow feature extraction.

        Returns:
            The predicted :class:`MarketRegime` for the most recent step.

        Raises:
            ModelError: If :meth:`fit` has not been called yet.
            DataFetchError: If the window is too short.
        """
        if self._model is None:
            raise ModelError(
                "HMMRegimeDetector has not been fitted.  Call fit() first."
            )

        prices = np.asarray(recent_window, dtype=np.float64)
        features = _compute_features(prices)

        hidden_states = self._model.predict(features)
        latest_state = int(hidden_states[-1])
        regime = self._state_to_regime.get(latest_state, MarketRegime.SIDEWAYS)

        log.debug(
            "regime_detector.predict",
            hidden_state=latest_state,
            regime=regime.name,
        )
        return regime


# ---------------------------------------------------------------------------
# Regime-aware weight presets
# ---------------------------------------------------------------------------

# Default per-agent weights; order matches [AgentD3QN, AgentDoubleDQN,
# AgentDoubleDQN, AgentTwinD3QN] from task1_ensemble.py.
_REGIME_WEIGHTS: dict[MarketRegime, list[float]] = {
    MarketRegime.BULL: [1.5, 0.8, 0.8, 1.0],       # D3QN more aggressive
    MarketRegime.BEAR: [0.7, 1.3, 1.3, 0.9],        # conservative agents up
    MarketRegime.SIDEWAYS: [1.0, 1.0, 1.0, 1.0],    # balanced
    MarketRegime.HIGH_VOLATILITY: [0.5, 0.5, 0.5, 0.5],  # reduced overall
}

# Scale factor applied to position sizes under HIGH_VOLATILITY
_HIGH_VOL_POSITION_SCALE = 0.5


# ---------------------------------------------------------------------------
# RegimeAwareEnsemble
# ---------------------------------------------------------------------------


class RegimeAwareEnsemble:
    """Wraps the voting ensemble and adjusts agent weights per detected regime.

    Calls :class:`HMMRegimeDetector` each cycle to obtain the current regime
    and modifies the vote weights passed to the ensemble accordingly.

    Args:
        detector: Fitted :class:`HMMRegimeDetector`.
        regime_weights: Optional override for per-regime weight vectors.
            Keys are :class:`MarketRegime`, values are lists of floats (one
            per agent, same order as ``agent_classes`` in the ensemble).
        position_size_base: Base position size used before regime scaling.

    Example::

        from regime_detector import HMMRegimeDetector, RegimeAwareEnsemble
        detector = HMMRegimeDetector().fit(train_prices)
        ensemble = RegimeAwareEnsemble(detector=detector)
        weights, size = ensemble.get_weights_and_size(recent_prices)
    """

    def __init__(
        self,
        detector: HMMRegimeDetector,
        regime_weights: dict[MarketRegime, list[float]] | None = None,
        position_size_base: float = 0.001,
    ) -> None:
        self.detector = detector
        self.regime_weights = regime_weights or _REGIME_WEIGHTS
        self.position_size_base = position_size_base
        self._current_regime: MarketRegime = MarketRegime.SIDEWAYS

        log.info(
            "regime_aware_ensemble.init",
            position_size_base=position_size_base,
        )

    @property
    def current_regime(self) -> MarketRegime:
        """The regime identified at the most recent :meth:`get_weights_and_size` call."""
        return self._current_regime

    def get_weights_and_size(
        self, recent_prices: npt.ArrayLike
    ) -> tuple[list[float], float]:
        """Compute ensemble vote weights and position size for the current regime.

        Args:
            recent_prices: Recent price window passed to the detector.

        Returns:
            A tuple ``(weights, position_size)`` where *weights* is a list of
            per-agent vote multipliers and *position_size* is the recommended
            trade quantity (scaled down under HIGH_VOLATILITY).
        """
        try:
            regime = self.detector.predict(recent_prices)
        except (ModelError, DataFetchError) as exc:
            log.warning(
                "regime_aware_ensemble.predict_failed",
                error=str(exc),
                fallback=MarketRegime.SIDEWAYS.name,
            )
            regime = MarketRegime.SIDEWAYS

        self._current_regime = regime
        weights = list(self.regime_weights.get(regime, [1.0, 1.0, 1.0, 1.0]))

        if regime == MarketRegime.HIGH_VOLATILITY:
            position_size = self.position_size_base * _HIGH_VOL_POSITION_SCALE
        else:
            position_size = self.position_size_base

        log.info(
            "regime_aware_ensemble.weights",
            regime=regime.name,
            weights=weights,
            position_size=round(position_size, 8),
        )
        return weights, position_size

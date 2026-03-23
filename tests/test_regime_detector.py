"""Tests for regime_detector.py."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import DataFetchError, ModelError
from regime_detector import (
    HMMRegimeDetector,
    MarketRegime,
    RegimeAwareEnsemble,
    _compute_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_bull_prices(n: int = 200) -> np.ndarray:
    """Generate a simple upward trending price series."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.005, n)
    prices = 30_000.0 * np.exp(np.cumsum(returns))
    return prices


def _synthetic_random_prices(n: int = 300) -> np.ndarray:
    """Generate a random-walk price series with mixed regimes."""
    rng = np.random.default_rng(0)
    segment_size = n // 4
    segments = []
    # Bull
    segments.append(rng.normal(0.002, 0.004, segment_size))
    # Bear
    segments.append(rng.normal(-0.002, 0.006, segment_size))
    # Sideways
    segments.append(rng.normal(0.0, 0.002, segment_size))
    # High vol
    segments.append(rng.normal(0.0, 0.025, n - 3 * segment_size))
    returns = np.concatenate(segments)
    return 30_000.0 * np.exp(np.cumsum(returns))


# ---------------------------------------------------------------------------
# MarketRegime
# ---------------------------------------------------------------------------


class TestMarketRegime:
    def test_all_four_regimes_exist(self) -> None:
        names = {r.name for r in MarketRegime}
        assert names == {"BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"}

    def test_enum_values_are_unique(self) -> None:
        values = [r.value for r in MarketRegime]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# _compute_features
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    def test_output_shape(self) -> None:
        prices = _synthetic_bull_prices(100)
        feats = _compute_features(prices)
        # N-1 returns, drop first 20 for rolling warm-up
        assert feats.shape[1] == 4
        assert feats.shape[0] == len(prices) - 1 - 20

    def test_short_series_raises(self) -> None:
        prices = np.arange(20, dtype=np.float64)
        with pytest.raises(DataFetchError):
            _compute_features(prices)

    def test_no_nans_in_normal_prices(self) -> None:
        prices = _synthetic_bull_prices(150)
        feats = _compute_features(prices)
        assert not np.isnan(feats).any()

    def test_2d_input_raises(self) -> None:
        prices = np.ones((50, 2))
        with pytest.raises(DataFetchError):
            _compute_features(prices)


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------


class TestHMMRegimeDetector:
    """Tests with a mocked hmmlearn model to avoid the heavy dependency."""

    def _mock_hmm(self, n_states: int = 4) -> MagicMock:
        """Build a lightweight mock GaussianHMM."""
        model = MagicMock()

        def _predict(X):
            # Cycle through states deterministically
            n = len(X)
            return np.array([i % n_states for i in range(n)])

        model.predict.side_effect = _predict
        return model

    @patch("regime_detector.HMMRegimeDetector._build_model")
    def test_fit_returns_self(self, mock_build: MagicMock) -> None:
        mock_build.return_value = self._mock_hmm()
        detector = HMMRegimeDetector()
        prices = _synthetic_random_prices(200)
        result = detector.fit(prices)
        assert result is detector

    @patch("regime_detector.HMMRegimeDetector._build_model")
    def test_predict_returns_market_regime(self, mock_build: MagicMock) -> None:
        mock_build.return_value = self._mock_hmm()
        detector = HMMRegimeDetector()
        prices = _synthetic_random_prices(200)
        detector.fit(prices)
        regime = detector.predict(prices[-80:])
        assert isinstance(regime, MarketRegime)

    @patch("regime_detector.HMMRegimeDetector._build_model")
    def test_state_mapping_covers_all_states(self, mock_build: MagicMock) -> None:
        mock_build.return_value = self._mock_hmm()
        detector = HMMRegimeDetector()
        prices = _synthetic_random_prices(300)
        detector.fit(prices)
        assert len(detector._state_to_regime) == 4
        regimes = set(detector._state_to_regime.values())
        assert len(regimes) >= 3  # at least 3 distinct regimes assigned

    def test_predict_before_fit_raises_model_error(self) -> None:
        detector = HMMRegimeDetector()
        with pytest.raises(ModelError, match="not been fitted"):
            detector.predict(np.ones(100))

    @patch("regime_detector.HMMRegimeDetector._build_model")
    def test_predict_too_short_raises(self, mock_build: MagicMock) -> None:
        mock_build.return_value = self._mock_hmm()
        detector = HMMRegimeDetector()
        prices = _synthetic_random_prices(200)
        detector.fit(prices)
        with pytest.raises(DataFetchError):
            detector.predict(np.ones(10))  # too short for feature extraction

    def test_missing_hmmlearn_raises_model_error(self) -> None:
        detector = HMMRegimeDetector()
        with patch.dict("sys.modules", {"hmmlearn": None, "hmmlearn.hmm": None}):
            with pytest.raises(ModelError, match="hmmlearn"):
                detector._build_model()


# ---------------------------------------------------------------------------
# RegimeAwareEnsemble
# ---------------------------------------------------------------------------


class TestRegimeAwareEnsemble:
    def _make_fitted_detector(self) -> HMMRegimeDetector:
        detector = HMMRegimeDetector()
        detector._model = MagicMock()
        detector._state_to_regime = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR,
            2: MarketRegime.SIDEWAYS,
            3: MarketRegime.HIGH_VOLATILITY,
        }
        # Always predict BULL (state 0)
        detector._model.predict.return_value = np.zeros(80, dtype=int)
        return detector

    def test_get_weights_bull_returns_d3qn_heavy(self) -> None:
        detector = self._make_fitted_detector()
        ensemble = RegimeAwareEnsemble(detector=detector)
        prices = _synthetic_random_prices(200)
        weights, size = ensemble.get_weights_and_size(prices[-80:])
        # D3QN (index 0) should have higher weight in BULL
        assert weights[0] > weights[1]

    def test_high_vol_reduces_position_size(self) -> None:
        detector = self._make_fitted_detector()
        # Override to always return HIGH_VOLATILITY (state 3)
        detector._model.predict.return_value = np.full(80, 3, dtype=int)
        ensemble = RegimeAwareEnsemble(detector=detector, position_size_base=0.002)
        prices = _synthetic_random_prices(200)
        _, size = ensemble.get_weights_and_size(prices[-80:])
        assert size < 0.002

    def test_sideways_returns_balanced_weights(self) -> None:
        detector = self._make_fitted_detector()
        # Override to always return SIDEWAYS (state 2)
        detector._model.predict.return_value = np.full(80, 2, dtype=int)
        ensemble = RegimeAwareEnsemble(detector=detector)
        prices = _synthetic_random_prices(200)
        weights, _ = ensemble.get_weights_and_size(prices[-80:])
        assert all(w == 1.0 for w in weights)

    def test_predict_failure_falls_back_to_sideways(self) -> None:
        detector = MagicMock(spec=HMMRegimeDetector)
        detector.predict.side_effect = ModelError("fail")
        ensemble = RegimeAwareEnsemble(detector=detector)
        prices = _synthetic_random_prices(200)
        weights, _ = ensemble.get_weights_and_size(prices[-80:])
        assert ensemble.current_regime == MarketRegime.SIDEWAYS

    def test_current_regime_property_updated(self) -> None:
        detector = self._make_fitted_detector()
        ensemble = RegimeAwareEnsemble(detector=detector)
        prices = _synthetic_random_prices(200)
        ensemble.get_weights_and_size(prices[-80:])
        assert ensemble.current_regime == MarketRegime.BULL

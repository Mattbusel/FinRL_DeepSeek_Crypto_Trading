"""Tests that MIN_CONFIDENCE_THRESHOLD is hard-enforced on all signal columns.

These tests verify that signals below the threshold are discarded (return None)
rather than merely logged, for both sentiment and risk analysis functions.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import deepseek_signals as ds
from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOW_CONFIDENCE_SENTIMENT = {
    "sentiment_score": 4,
    "confidence_score_sentiment": 0.05,  # below default threshold 0.3
    "reasoning_sentiment": "Weak signal.",
}

_HIGH_CONFIDENCE_SENTIMENT = {
    "sentiment_score": 4,
    "confidence_score_sentiment": 0.95,
    "reasoning_sentiment": "Strong bullish signal.",
}

_LOW_CONFIDENCE_RISK = {
    "risk_score": 3,
    "confidence_score_risk": 0.05,  # below default threshold 0.3
    "reasoning_risk": "Weak risk signal.",
}

_HIGH_CONFIDENCE_RISK = {
    "risk_score": 3,
    "confidence_score_risk": 0.80,
    "reasoning_risk": "Clear risk environment.",
}

_BOUNDARY_CONFIDENCE_SENTIMENT = {
    "sentiment_score": 3,
    "confidence_score_sentiment": 0.3,  # exactly at threshold
    "reasoning_sentiment": "Borderline.",
}

_BOUNDARY_CONFIDENCE_RISK = {
    "risk_score": 3,
    "confidence_score_risk": 0.3,
    "reasoning_risk": "Borderline.",
}


# ---------------------------------------------------------------------------
# Sentiment confidence enforcement
# ---------------------------------------------------------------------------


class TestSentimentConfidenceEnforcement:
    """Verify analyze_article_sentiment discards low-confidence results."""

    def test_low_confidence_sentiment_returns_none(self) -> None:
        """Signal below MIN_CONFIDENCE_THRESHOLD must be discarded (None)."""
        with patch.object(ds, "_call_api", return_value=_LOW_CONFIDENCE_SENTIMENT):
            result = ds.analyze_article_sentiment("BTC news", "Some text.")
        assert result is None, (
            "analyze_article_sentiment should return None when confidence < threshold"
        )

    def test_high_confidence_sentiment_returns_data(self) -> None:
        """Signal above MIN_CONFIDENCE_THRESHOLD must pass through."""
        with patch.object(ds, "_call_api", return_value=_HIGH_CONFIDENCE_SENTIMENT):
            result = ds.analyze_article_sentiment("BTC news", "Some text.")
        assert result is not None
        assert result["sentiment_score"] == 4

    def test_boundary_confidence_sentiment_passes(self) -> None:
        """A signal exactly at the threshold is NOT discarded (>= check)."""
        with patch.object(ds, "_call_api", return_value=_BOUNDARY_CONFIDENCE_SENTIMENT):
            result = ds.analyze_article_sentiment("BTC news", "Some text.")
        assert result is not None

    def test_zero_confidence_sentiment_returns_none(self) -> None:
        """Zero confidence must be discarded unconditionally."""
        data = dict(_LOW_CONFIDENCE_SENTIMENT)
        data["confidence_score_sentiment"] = 0.0
        with patch.object(ds, "_call_api", return_value=data):
            result = ds.analyze_article_sentiment("BTC", "text")
        assert result is None

    def test_missing_confidence_key_returns_none(self) -> None:
        """Missing key defaults to 0.0 confidence, which is below threshold."""
        data = {"sentiment_score": 4, "reasoning_sentiment": "ok"}
        with patch.object(ds, "_call_api", return_value=data):
            result = ds.analyze_article_sentiment("BTC", "text")
        # Missing required key -> returns None from key validation, not confidence check
        assert result is None


# ---------------------------------------------------------------------------
# Risk confidence enforcement
# ---------------------------------------------------------------------------


class TestRiskConfidenceEnforcement:
    """Verify analyze_article_risk discards low-confidence results."""

    def test_low_confidence_risk_returns_none(self) -> None:
        """Risk signal below threshold must be discarded."""
        with patch.object(ds, "_call_api", return_value=_LOW_CONFIDENCE_RISK):
            result = ds.analyze_article_risk("BTC news", "Some text.")
        assert result is None, (
            "analyze_article_risk should return None when confidence < threshold"
        )

    def test_high_confidence_risk_returns_data(self) -> None:
        """Risk signal above threshold must pass through."""
        with patch.object(ds, "_call_api", return_value=_HIGH_CONFIDENCE_RISK):
            result = ds.analyze_article_risk("BTC news", "Some text.")
        assert result is not None
        assert result["risk_score"] == 3

    def test_boundary_confidence_risk_passes(self) -> None:
        """Risk signal exactly at threshold is NOT discarded."""
        with patch.object(ds, "_call_api", return_value=_BOUNDARY_CONFIDENCE_RISK):
            result = ds.analyze_article_risk("BTC news", "Some text.")
        assert result is not None

    def test_zero_confidence_risk_returns_none(self) -> None:
        """Zero confidence risk must be discarded."""
        data = dict(_LOW_CONFIDENCE_RISK)
        data["confidence_score_risk"] = 0.0
        with patch.object(ds, "_call_api", return_value=data):
            result = ds.analyze_article_risk("BTC", "text")
        assert result is None

    def test_missing_risk_confidence_key_returns_none(self) -> None:
        """Missing required key -> return None (validated by key check)."""
        data = {"risk_score": 3, "reasoning_risk": "ok"}
        with patch.object(ds, "_call_api", return_value=data):
            result = ds.analyze_article_risk("BTC", "text")
        assert result is None


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    """Verify threshold enforcement works for non-default threshold values."""

    def test_custom_high_threshold_discards_moderate_confidence(self) -> None:
        """A confidence of 0.5 should be discarded when threshold is 0.7."""
        original = settings.min_confidence_threshold
        settings.min_confidence_threshold = 0.7
        try:
            data = {
                "sentiment_score": 4,
                "confidence_score_sentiment": 0.5,
                "reasoning_sentiment": "ok",
            }
            with patch.object(ds, "_call_api", return_value=data):
                result = ds.analyze_article_sentiment("BTC", "text")
            assert result is None
        finally:
            settings.min_confidence_threshold = original

    def test_custom_low_threshold_accepts_previously_discarded(self) -> None:
        """A confidence of 0.2 passes when threshold is lowered to 0.1."""
        original = settings.min_confidence_threshold
        settings.min_confidence_threshold = 0.1
        try:
            data = {
                "sentiment_score": 3,
                "confidence_score_sentiment": 0.2,
                "reasoning_sentiment": "borderline ok",
            }
            with patch.object(ds, "_call_api", return_value=data):
                result = ds.analyze_article_sentiment("BTC", "text")
            assert result is not None
        finally:
            settings.min_confidence_threshold = original

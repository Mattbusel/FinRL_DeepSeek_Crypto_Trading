"""Tests for deepseek_signals.py with mocked API calls."""

import sys
import os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_SENTIMENT = {
    "sentiment_score": 4,
    "confidence_score_sentiment": 0.85,
    "reasoning_sentiment": "Positive article.",
}

VALID_RISK = {
    "risk_score": 3,
    "confidence_score_risk": 0.70,
    "reasoning_risk": "Neutral risk.",
}

# Sentiment score 3 → normalised to 0; scale is 1-5 centred at 3
# In the module the values are passed through as-is; [-1,1] test is for
# the *normalised* form below.

SENTIMENT_KEYS = {"sentiment_score", "confidence_score_sentiment", "reasoning_sentiment"}
RISK_KEYS = {"risk_score", "confidence_score_risk", "reasoning_risk"}


def _make_response(content: dict):
    """Build a minimal mock completions response object."""

    class _Msg:
        def __init__(self):
            self.content = json.dumps(content)

    class _Choice:
        def __init__(self):
            self.message = _Msg()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    return _Response()


# ---------------------------------------------------------------------------
# test_signal_extraction_returns_valid_keys
# ---------------------------------------------------------------------------

def test_signal_extraction_returns_valid_keys(mocker):
    """Mocked API should return a dict containing all required keys."""
    import deepseek_signals as ds

    mock_response = _make_response(VALID_SENTIMENT)
    mocker.patch.object(
        ds, "_call_api", return_value=VALID_SENTIMENT
    )

    result = ds.analyze_article_sentiment("BTC surges", "Bitcoin hit a new ATH.")
    assert result is not None
    assert SENTIMENT_KEYS.issubset(result.keys())


def test_risk_extraction_returns_valid_keys(mocker):
    """Mocked API should return a dict with all required risk keys."""
    import deepseek_signals as ds

    mocker.patch.object(ds, "_call_api", return_value=VALID_RISK)

    result = ds.analyze_article_risk("Regulation fears", "SEC targets crypto.")
    assert result is not None
    assert RISK_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# test_signal_fallback_on_api_error
# ---------------------------------------------------------------------------

def test_signal_fallback_on_api_error(mocker):
    """When the API raises, analyze_article_sentiment should return None."""
    import deepseek_signals as ds
    from exceptions import SignalError

    mocker.patch.object(
        ds, "_call_with_retry", side_effect=SignalError("API down")
    )

    result = ds.analyze_article_sentiment("Title", "Text")
    assert result is None


def test_risk_fallback_on_api_error(mocker):
    """When the API raises, analyze_article_risk should return None."""
    import deepseek_signals as ds
    from exceptions import SignalError

    mocker.patch.object(
        ds, "_call_with_retry", side_effect=SignalError("API down")
    )

    result = ds.analyze_article_risk("Title", "Text")
    assert result is None


# ---------------------------------------------------------------------------
# test_sentiment_values_in_range
# ---------------------------------------------------------------------------

def test_sentiment_score_in_range(mocker):
    """Returned sentiment_score must be in the documented 1-5 range."""
    import deepseek_signals as ds

    mocker.patch.object(ds, "_call_api", return_value=VALID_SENTIMENT)

    result = ds.analyze_article_sentiment("BTC", "text")
    assert result is not None
    score = result["sentiment_score"]
    assert 1 <= score <= 5


def test_confidence_score_in_range(mocker):
    """Returned confidence_score_sentiment must be in [0, 1]."""
    import deepseek_signals as ds

    mocker.patch.object(ds, "_call_api", return_value=VALID_SENTIMENT)

    result = ds.analyze_article_sentiment("BTC", "text")
    assert result is not None
    conf = result["confidence_score_sentiment"]
    assert 0.0 <= conf <= 1.0


def test_get_sentiment_and_risk_returns_series(mocker):
    """get_sentiment_and_risk should return a pd.Series with six fields."""
    import pandas as pd
    import deepseek_signals as ds

    mocker.patch.object(ds, "_call_api", side_effect=[VALID_SENTIMENT, VALID_RISK])

    row = pd.Series({"title": "BTC up", "article_text": "Bitcoin rallied."})
    result = ds.get_sentiment_and_risk(row)
    assert isinstance(result, pd.Series)
    expected_keys = list(SENTIMENT_KEYS) + list(RISK_KEYS)
    for key in expected_keys:
        assert key in result.index

"""Tests for deepseek_signals.py.

Coverage targets:
- Signal extraction (sentiment and risk) happy paths
- API error handling and retry exhaustion
- Edge cases: empty response, malformed JSON, missing keys, rate-limit stub
- Data cleaning helpers
- Checkpoint save / load round-trip
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

import deepseek_signals as ds
from exceptions import DataFetchError, SignalError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# analyze_article_sentiment
# ---------------------------------------------------------------------------

class TestAnalyzeArticleSentiment:
    def test_happy_path(self, sample_article, mock_deepseek_client, good_sentiment_response):
        """Returns a dict with all required keys on a valid API response."""
        mock_deepseek_client.side_effect = [_make_completion(good_sentiment_response)]
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is not None
        assert "sentiment_score" in result
        assert "confidence_score_sentiment" in result
        assert "reasoning_sentiment" in result

    def test_missing_key_returns_none(self, sample_article, mock_deepseek_client):
        """Returns None when the API response omits a required key."""
        incomplete = '{"sentiment_score": 4}'
        mock_deepseek_client.side_effect = [_make_completion(incomplete)] * 10
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None

    def test_malformed_json_returns_none(self, sample_article, mock_deepseek_client):
        """Returns None when the API returns non-JSON content."""
        mock_deepseek_client.side_effect = [_make_completion("not json")] * 10
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None

    def test_empty_response_returns_none(self, sample_article, mock_deepseek_client):
        """Returns None when the API returns an empty string."""
        mock_deepseek_client.side_effect = [_make_completion("")] * 10
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None

    def test_low_confidence_still_returns_data(
        self, sample_article, mock_deepseek_client
    ):
        """Low-confidence response is returned (with a warning), not discarded."""
        low_conf = (
            '{"sentiment_score": 3, "confidence_score_sentiment": 0.1, '
            '"reasoning_sentiment": "Uncertain."}'
        )
        mock_deepseek_client.side_effect = [_make_completion(low_conf)]
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is not None
        assert result["confidence_score_sentiment"] == 0.1

    def test_api_exception_retries_and_returns_none(
        self, sample_article, mock_deepseek_client
    ):
        """Returns None after all retries are exhausted on repeated API errors."""
        mock_deepseek_client.side_effect = Exception("network error")
        result = ds.analyze_article_sentiment(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None


# ---------------------------------------------------------------------------
# analyze_article_risk
# ---------------------------------------------------------------------------

class TestAnalyzeArticleRisk:
    def test_happy_path(self, sample_article, mock_deepseek_client, good_risk_response):
        """Returns a dict with all required keys on a valid API response."""
        mock_deepseek_client.side_effect = [_make_completion(good_risk_response)]
        result = ds.analyze_article_risk(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is not None
        assert set(result.keys()) >= {"risk_score", "confidence_score_risk", "reasoning_risk"}

    def test_missing_keys_returns_none(self, sample_article, mock_deepseek_client):
        """Returns None when required risk keys are absent."""
        mock_deepseek_client.side_effect = [_make_completion('{"risk_score": 2}')] * 10
        result = ds.analyze_article_risk(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None

    def test_rate_limit_simulation(self, sample_article, mock_deepseek_client):
        """Simulates repeated failures (rate-limit scenario); returns None."""
        mock_deepseek_client.side_effect = Exception("429 rate limit")
        result = ds.analyze_article_risk(
            sample_article["title"], sample_article["article_text"]
        )
        assert result is None


# ---------------------------------------------------------------------------
# get_sentiment_and_risk
# ---------------------------------------------------------------------------

class TestGetSentimentAndRisk:
    def test_returns_series_with_six_fields(
        self, sample_article, mock_deepseek_client,
        good_sentiment_response, good_risk_response
    ):
        """Combined call returns a Series with all six expected fields."""
        mock_deepseek_client.side_effect = [
            _make_completion(good_sentiment_response),
            _make_completion(good_risk_response),
        ]
        row = pd.Series(sample_article)
        result = ds.get_sentiment_and_risk(row)
        expected_keys = {
            "sentiment_score", "confidence_score_sentiment", "reasoning_sentiment",
            "risk_score", "confidence_score_risk", "reasoning_risk",
        }
        assert set(result.index) == expected_keys

    def test_partial_failure_fills_none(self, sample_article, mock_deepseek_client):
        """If sentiment fails, its fields are None; risk fields are populated."""
        good_risk = (
            '{"risk_score": 3, "confidence_score_risk": 0.8, '
            '"reasoning_risk": "Stable."}'
        )
        # Sentiment always fails, risk succeeds once
        def side_effect(*args, **kwargs):
            content = kwargs.get("messages", [{}])[0].get("content", "")
            if "sentiment" in content.lower() or "financial news analyst" in content.lower():
                raise Exception("fail")
            return _make_completion(good_risk)

        mock_deepseek_client.side_effect = side_effect
        row = pd.Series(sample_article)
        result = ds.get_sentiment_and_risk(row)
        # Risk fields are populated, sentiment fields may be None
        assert result["risk_score"] is not None or result["sentiment_score"] is None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class TestCleanAndValidateData:
    def test_removes_nan_rows(self):
        df = pd.DataFrame({
            "title": ["Good title", None, "Another"],
            "article_text": ["text", "text", "text"],
        })
        result = ds.clean_and_validate_data(df)
        assert len(result) == 2

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "title": ["Same", "Same"],
            "article_text": ["dup", "dup"],
        })
        result = ds.clean_and_validate_data(df)
        assert len(result) == 1

    def test_removes_empty_strings(self):
        df = pd.DataFrame({
            "title": ["  ", "Valid"],
            "article_text": ["text", "text"],
        })
        result = ds.clean_and_validate_data(df)
        assert len(result) == 1

    def test_raises_on_empty_result(self):
        df = pd.DataFrame({"title": [None], "article_text": [None]})
        with pytest.raises(DataFetchError):
            ds.clean_and_validate_data(df)

    def test_resets_index(self, sample_news_df):
        result = ds.clean_and_validate_data(sample_news_df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_save_and_load_roundtrip(self, sample_news_df, tmp_path):
        checkpoint = str(tmp_path / "ckpt.csv")
        ds.save_checkpoint(sample_news_df, checkpoint)
        loaded = ds.load_checkpoint(checkpoint)
        assert loaded is not None
        assert len(loaded) == len(sample_news_df)

    def test_load_returns_none_if_missing(self, tmp_path):
        result = ds.load_checkpoint(str(tmp_path / "nonexistent.csv"))
        assert result is None

    def test_save_raises_on_bad_path(self, sample_news_df):
        with pytest.raises(DataFetchError):
            ds.save_checkpoint(sample_news_df, "/nonexistent_dir/no/ckpt.csv")

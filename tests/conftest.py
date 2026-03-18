"""Shared pytest fixtures and mocks for the LARSA test suite.

Provides:
- ``mock_deepseek_client``: replaces the module-level OpenAI client so tests
  never hit the real DeepSeek API.
- ``sample_article``: a small dict with ``title`` and ``article_text``.
- ``sample_news_df``: a three-row DataFrame suitable for pipeline tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(content: str) -> MagicMock:
    """Build a minimal mock object that looks like an OpenAI ChatCompletion."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_article() -> dict[str, str]:
    """A minimal article dict with title and text fields."""
    return {
        "title": "Bitcoin hits all-time high as institutional demand surges",
        "article_text": (
            "Bitcoin reached a new record price today driven by significant "
            "institutional buying pressure and ETF inflows."
        ),
    }


@pytest.fixture()
def sample_news_df() -> pd.DataFrame:
    """A three-row DataFrame with valid title and article_text columns."""
    return pd.DataFrame(
        {
            "title": [
                "Bitcoin rallies on positive macro data",
                "Crypto exchange faces regulatory scrutiny",
                "DeFi protocol launches new yield product",
            ],
            "article_text": [
                "BTC rose 5% after better-than-expected CPI figures.",
                "A major exchange received a formal notice from the SEC.",
                "A new DeFi protocol promises 20% APY on stablecoin deposits.",
            ],
        }
    )


@pytest.fixture()
def mock_deepseek_client():
    """Patch the module-level _client in deepseek_signals to avoid real API calls.

    Yields the mocked ``chat.completions.create`` callable so individual
    tests can configure return values.
    """
    sentiment_payload = (
        '{"sentiment_score": 4, '
        '"confidence_score_sentiment": 0.9, '
        '"reasoning_sentiment": "Positive outlook."}'
    )
    risk_payload = (
        '{"risk_score": 3, '
        '"confidence_score_risk": 0.8, '
        '"reasoning_risk": "Neutral risk environment."}'
    )

    client_mock = MagicMock()
    # By default alternate between sentiment and risk payloads.
    client_mock.chat.completions.create.side_effect = [
        _make_completion(sentiment_payload),
        _make_completion(risk_payload),
    ] * 20  # enough for any single test

    import deepseek_signals as ds

    with patch.object(ds, "_client", client_mock):
        yield client_mock.chat.completions.create


@pytest.fixture()
def good_sentiment_response() -> str:
    """Valid JSON sentiment response string."""
    return (
        '{"sentiment_score": 4, '
        '"confidence_score_sentiment": 0.9, '
        '"reasoning_sentiment": "Strong bullish signals."}'
    )


@pytest.fixture()
def good_risk_response() -> str:
    """Valid JSON risk response string."""
    return (
        '{"risk_score": 2, '
        '"confidence_score_risk": 0.75, '
        '"reasoning_risk": "Moderate regulatory risk."}'
    )

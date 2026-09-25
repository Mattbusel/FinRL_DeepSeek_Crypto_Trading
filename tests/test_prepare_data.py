"""Tests for scripts/prepare_data.py: neutral default and news merge."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prepare_data import NEUTRAL, attach_signals  # noqa: E402


def _seconds(n: int) -> pd.DataFrame:
    t = pd.date_range("2021-04-14 04:00:00", periods=n, freq="s", tz="UTC")
    return pd.DataFrame({"system_time": t.astype(str), "midpoint": 60000.0})


def test_no_news_is_neutral():
    df = attach_signals(_seconds(5), None)
    assert (df["sentiment_score"] == NEUTRAL).all()
    assert (df["risk_score"] == NEUTRAL).all()


def test_news_is_forward_filled_from_its_timestamp(tmp_path):
    news = pd.DataFrame(
        {
            "date": ["2021-04-14 04:00:02+00:00"],
            "title": ["x"],
            "sentiment_score": [5],
            "risk_score": [2],
        }
    )
    path = tmp_path / "news.csv"
    news.to_csv(path, index=False)
    df = attach_signals(_seconds(5), str(path))
    assert df["sentiment_score"].tolist() == [3.0, 3.0, 5.0, 5.0, 5.0]
    assert df["risk_score"].tolist() == [3.0, 3.0, 2.0, 2.0, 2.0]

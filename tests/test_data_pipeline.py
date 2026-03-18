"""Tests for data utilities: normalization, cleaning, and checkpoint helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepseek_signals import clean_and_validate_data, load_checkpoint, save_checkpoint
from exceptions import DataFetchError


def _make_df(**overrides) -> pd.DataFrame:
    data = {
        "title": ["BTC rallies", "ETH dips", "Market update"],
        "article_text": ["Text A", "Text B", "Text C"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestCleanAndValidateData:
    def test_removes_null_title(self):
        df = _make_df(title=["Good title", None, "Another"])
        assert len(clean_and_validate_data(df)) == 2

    def test_removes_null_article_text(self):
        df = _make_df(article_text=["Text", None, "More"])
        assert len(clean_and_validate_data(df)) == 2

    def test_removes_duplicates(self):
        df = pd.DataFrame(
            {
                "title": ["Same", "Same", "Unique"],
                "article_text": ["Body", "Body", "Other body"],
            }
        )
        assert len(clean_and_validate_data(df)) == 2

    def test_removes_empty_string_title(self):
        df = _make_df(title=["", "Valid", "Another"])
        assert len(clean_and_validate_data(df)) == 2

    def test_removes_whitespace_only_title(self):
        df = _make_df(title=["   ", "Real title", "Another"])
        assert len(clean_and_validate_data(df)) == 2

    def test_resets_index(self):
        result = clean_and_validate_data(_make_df())
        assert list(result.index) == list(range(len(result)))

    def test_raises_when_all_invalid(self):
        df = pd.DataFrame({"title": [None, None], "article_text": [None, None]})
        with pytest.raises(DataFetchError, match="No valid rows"):
            clean_and_validate_data(df)

    def test_valid_rows_preserved(self):
        assert len(clean_and_validate_data(_make_df())) == 3


class TestCheckpointRoundtrip:
    def test_save_and_load(self, tmp_path):
        df = _make_df()
        path = str(tmp_path / "checkpoint.csv")
        save_checkpoint(df, path)
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert len(loaded) == len(df)

    def test_load_returns_none_for_missing_file(self, tmp_path):
        assert load_checkpoint(str(tmp_path / "missing.csv")) is None

    def test_save_raises_on_bad_path(self):
        with pytest.raises(DataFetchError):
            save_checkpoint(_make_df(), "/nonexistent_xyz/out.csv")


class TestNormalization:
    @staticmethod
    def z_score(arr: np.ndarray) -> np.ndarray:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return (arr - mean) / std

    def test_zero_mean(self):
        data = np.random.default_rng(0).standard_normal((100, 5))
        np.testing.assert_allclose(self.z_score(data).mean(axis=0), 0.0, atol=1e-10)

    def test_unit_variance(self):
        data = np.random.default_rng(1).standard_normal((100, 5))
        np.testing.assert_allclose(self.z_score(data).std(axis=0), 1.0, atol=1e-10)

    def test_constant_column_handled(self):
        assert np.all(self.z_score(np.ones((50, 3))) == 0.0)

    def test_shape_preserved(self):
        data = np.random.randn(20, 8)
        assert self.z_score(data).shape == data.shape

    def test_missing_data_imputed(self):
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        col_means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = col_means[inds[1]]
        assert not np.isnan(data).any()

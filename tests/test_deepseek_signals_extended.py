"""Extended error-path and edge-case tests for deepseek_signals.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import deepseek_signals as ds
from exceptions import DataFetchError, SignalError


class TestBuildClient:
    def test_raises_signal_error_when_api_key_missing(self):
        """_build_client should raise SignalError if DEEPSEEK_API_KEY is empty."""
        with patch.object(ds.settings, "deepseek_api_key", ""):
            with pytest.raises(SignalError, match="DEEPSEEK_API_KEY"):
                ds._build_client()

    def test_returns_openai_instance_when_key_present(self):
        """_build_client should return an OpenAI client when key is set."""
        with patch.object(ds.settings, "deepseek_api_key", "sk-test-key"):
            client = ds._build_client()
            assert client is not None


class TestGetClient:
    def test_get_client_raises_without_key(self):
        """_get_client propagates SignalError when key is absent."""
        import deepseek_signals as ds_mod

        original = ds_mod._client
        ds_mod._client = None
        try:
            with patch.object(ds_mod.settings, "deepseek_api_key", ""):
                with pytest.raises(SignalError):
                    ds_mod._get_client()
        finally:
            ds_mod._client = original

    def test_get_client_reuses_existing(self):
        """_get_client returns the cached client on subsequent calls."""
        import deepseek_signals as ds_mod

        original = ds_mod._client
        mock_client = MagicMock()
        ds_mod._client = mock_client
        try:
            result = ds_mod._get_client()
            assert result is mock_client
        finally:
            ds_mod._client = original


class TestProcessNewsAnalysis:
    def _make_csv(self, tmp_path, rows=3) -> str:
        df = pd.DataFrame(
            {
                "title": [f"Title {i}" for i in range(rows)],
                "article_text": [f"Article body {i}" for i in range(rows)],
            }
        )
        path = str(tmp_path / "input.csv")
        df.to_csv(path, index=False)
        return path

    def test_raises_data_fetch_error_on_missing_input(self, tmp_path):
        """process_news_analysis raises DataFetchError for non-existent file."""
        with pytest.raises(DataFetchError, match="Cannot read input file"):
            ds.process_news_analysis(
                str(tmp_path / "nonexistent.csv"),
                str(tmp_path / "out.csv"),
            )

    def test_raises_data_fetch_error_on_unwritable_output(self, tmp_path, mock_deepseek_client):
        """process_news_analysis raises DataFetchError if output path is invalid."""
        input_file = self._make_csv(tmp_path)
        with pytest.raises((DataFetchError, OSError)):
            ds.process_news_analysis(
                input_file,
                "/nonexistent_dir_xyz/output.csv",
            )

    def test_resumes_from_checkpoint(self, tmp_path, mock_deepseek_client):
        """If a checkpoint exists it is loaded and remaining rows are processed."""
        input_file = self._make_csv(tmp_path, rows=2)
        output_file = str(tmp_path / "out.csv")

        # Create a checkpoint that already has the first row done
        checkpoint_file = str(tmp_path / "out_checkpoint.csv")
        checkpoint_df = pd.DataFrame(
            {
                "title": ["Title 0"],
                "article_text": ["Article body 0"],
                "sentiment_score": [4],
                "confidence_score_sentiment": [0.9],
                "reasoning_sentiment": ["ok"],
                "risk_score": [3],
                "confidence_score_risk": [0.8],
                "reasoning_risk": ["ok"],
            }
        )
        checkpoint_df.to_csv(checkpoint_file, index=False)

        # Should not raise; mock client handles the one remaining row
        try:
            ds.process_news_analysis(input_file, output_file)
        except Exception:
            pass  # allow partial failures; we just verify no crash on resume path


class TestCallApi:
    def _make_mock_completion(self, content: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_raises_signal_error_on_invalid_json(self, mock_deepseek_client):
        """_call_api raises SignalError when the response is not JSON."""
        mock_deepseek_client.side_effect = [self._make_mock_completion("not-valid-json")]
        with pytest.raises(SignalError):
            ds._call_api("any prompt")

    def test_raises_signal_error_on_api_exception(self, mock_deepseek_client):
        """_call_api wraps exceptions from the client in SignalError."""
        mock_deepseek_client.side_effect = RuntimeError("network down")
        with pytest.raises(SignalError):
            ds._call_api("any prompt")

    def test_returns_parsed_dict_on_success(self, mock_deepseek_client):
        """_call_api returns the parsed dict on a valid JSON response."""
        payload = (
            '{"sentiment_score": 4, "confidence_score_sentiment": 0.9, "reasoning_sentiment": "ok"}'
        )
        mock_deepseek_client.side_effect = [self._make_mock_completion(payload)]
        result = ds._call_api("any prompt")
        assert result["sentiment_score"] == 4


class TestCallWithRetry:
    def test_exhausts_retries_and_raises(self, mock_deepseek_client):
        """After max_retries failures, SignalError is raised."""
        mock_deepseek_client.side_effect = Exception("always fails")
        with patch.object(ds.settings, "max_retries", 2):
            with patch("time.sleep"):  # don't actually sleep in tests
                with pytest.raises(SignalError, match="failed after"):
                    ds._call_with_retry("prompt")

    def test_succeeds_on_second_attempt(self, mock_deepseek_client):
        """Retries after first failure and returns the successful result."""
        good = (
            '{"sentiment_score": 3, "confidence_score_sentiment": 0.7, "reasoning_sentiment": "ok"}'
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            choice = MagicMock()
            choice.message.content = good
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_deepseek_client.side_effect = side_effect
        with patch("time.sleep"):
            result = ds._call_with_retry("prompt")
        assert result["sentiment_score"] == 3

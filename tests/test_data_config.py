"""Tests for data_config.py (ConfigData) and seq_data.ConfigData."""

from __future__ import annotations

import pytest

from data_config import ConfigData


class TestDataConfigPaths:
    def test_default_data_dir(self):
        cfg = ConfigData()
        assert cfg.data_dir == "./data"

    def test_custom_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/mydata")
        assert cfg.data_dir == "/tmp/mydata"

    def test_csv_path_under_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/d")
        assert cfg.csv_path.startswith("/tmp/d")

    def test_input_ary_path_under_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/d")
        assert cfg.input_ary_path.startswith("/tmp/d")

    def test_label_ary_path_under_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/d")
        assert cfg.label_ary_path.startswith("/tmp/d")

    def test_predict_net_path_under_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/d")
        assert cfg.predict_net_path.startswith("/tmp/d")

    def test_predict_ary_path_under_data_dir(self):
        cfg = ConfigData(data_dir="/tmp/d")
        assert cfg.predict_ary_path.startswith("/tmp/d")

    def test_paths_are_strings(self):
        cfg = ConfigData()
        for attr in (
            "csv_path",
            "input_ary_path",
            "label_ary_path",
            "predict_net_path",
            "predict_ary_path",
        ):
            assert isinstance(getattr(cfg, attr), str), f"{attr} should be str"

    def test_csv_path_has_expected_filename(self):
        cfg = ConfigData()
        assert "BTC_1sec_with_sentiment_risk_train.csv" in cfg.csv_path

    def test_input_ary_has_npy_extension(self):
        cfg = ConfigData()
        assert cfg.input_ary_path.endswith(".npy")

    def test_label_ary_has_npy_extension(self):
        cfg = ConfigData()
        assert cfg.label_ary_path.endswith(".npy")

    def test_predict_net_has_pth_extension(self):
        cfg = ConfigData()
        assert cfg.predict_net_path.endswith(".pth")

    def test_predict_ary_has_npy_extension(self):
        cfg = ConfigData()
        assert cfg.predict_ary_path.endswith(".npy")

    def test_non_string_data_dir_raises_type_error(self):
        with pytest.raises(TypeError, match="data_dir must be a str"):
            ConfigData(data_dir=123)  # type: ignore[arg-type]

    def test_none_data_dir_raises_type_error(self):
        with pytest.raises(TypeError):
            ConfigData(data_dir=None)  # type: ignore[arg-type]


class TestSeqDataConfigData:
    """Tests for the ConfigData in seq_data (legacy, no type guard)."""

    def test_paths_under_data_dir(self):
        from seq_data import ConfigData as SeqConfigData  # type: ignore

        cfg = SeqConfigData(data_dir="/tmp/seq")
        for attr in (
            "csv_path",
            "input_ary_path",
            "label_ary_path",
            "predict_ary_path",
            "predict_net_path",
        ):
            val = getattr(cfg, attr, None)
            if val is not None:
                assert "/tmp/seq" in val, f"{attr} should be under /tmp/seq"

"""Tests for seq_run.py.

Coverage targets:
- SeqData initialisation and data splitting
- SeqData.sample_for_train returns correctly shaped tensors
- Graceful handling of missing input/label arrays
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

th = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# SeqData tests
# ---------------------------------------------------------------------------


class TestSeqData:
    """Tests for the SeqData data-loading and batching class."""

    def _make_fake_args(self, tmp_path) -> MagicMock:
        """Create a minimal ConfigData-like mock pointing at temporary files."""
        input_path = str(tmp_path / "input.npy")
        label_path = str(tmp_path / "label.npy")

        # 2000 time steps, 10 input features, 4 label channels
        np.save(input_path, np.random.randn(2000, 10).astype(np.float32))
        np.save(label_path, np.random.randn(2000, 4).astype(np.float32))

        args = MagicMock()
        args.input_ary_path = input_path
        args.label_ary_path = label_path
        return args

    def test_dimensions_are_set_correctly(self, tmp_path):
        """input_dim and label_dim match the saved array column counts."""
        from seq_run import SeqData

        args = self._make_fake_args(tmp_path)
        seq_data = SeqData(args=args, train_ratio=0.8)

        assert seq_data.input_dim == 10
        assert seq_data.label_dim == 4

    def test_train_valid_split(self, tmp_path):
        """Training and validation lengths respect the requested ratio."""
        from seq_run import SeqData

        args = self._make_fake_args(tmp_path)
        seq_data = SeqData(args=args, train_ratio=0.6)

        total = seq_data.train_seq_len + seq_data.valid_seq_len
        # Total should be close to 2000 - 1024 (seq_i2 formula)
        assert total > 0
        assert seq_data.train_seq_len > 0
        assert seq_data.valid_seq_len >= 0

    def test_sample_for_train_shape(self, tmp_path):
        """sample_for_train returns tensors with expected shapes."""
        from seq_run import SeqData

        # Need enough data for a meaningful batch
        input_path = str(tmp_path / "input.npy")
        label_path = str(tmp_path / "label.npy")
        np.save(input_path, np.random.randn(8000, 10).astype(np.float32))
        np.save(label_path, np.random.randn(8000, 4).astype(np.float32))

        args = MagicMock()
        args.input_ary_path = input_path
        args.label_ary_path = label_path

        seq_data = SeqData(args=args, train_ratio=0.8)
        batch_size = 4
        seq_len = 32
        inp, lab = seq_data.sample_for_train(
            batch_size=batch_size, seq_len=seq_len, device=th.device("cpu")
        )

        # Shape: (seq_len, batch_size, feature_dim)
        assert inp.shape == (seq_len, batch_size, seq_data.input_dim)
        assert lab.shape == (seq_len, batch_size, seq_data.label_dim)

    def test_data_generation_triggered_when_files_missing(self, tmp_path):
        """convert_btc_csv_to_btc_npy is called when arrays are absent."""
        from seq_run import SeqData

        args = MagicMock()
        args.input_ary_path = str(tmp_path / "missing_input.npy")
        args.label_ary_path = str(tmp_path / "missing_label.npy")

        dummy_input = np.random.randn(2000, 10).astype(np.float32)
        dummy_label = np.random.randn(2000, 4).astype(np.float32)

        def fake_convert(args):
            np.save(args.input_ary_path, dummy_input)
            np.save(args.label_ary_path, dummy_label)

        with patch("seq_run.convert_btc_csv_to_btc_npy", side_effect=fake_convert) as mock_fn:
            seq_data = SeqData(args=args, train_ratio=0.8)
            mock_fn.assert_called_once_with(args=args)
            assert seq_data.input_dim == 10

    def test_nan_in_input_is_replaced(self, tmp_path):
        """NaN values in the input array are replaced with zeros."""
        input_path = str(tmp_path / "input_nan.npy")
        label_path = str(tmp_path / "label_nan.npy")

        inp_arr = np.random.randn(2000, 10).astype(np.float32)
        inp_arr[0, 0] = float("nan")
        np.save(input_path, inp_arr)
        np.save(label_path, np.random.randn(2000, 4).astype(np.float32))

        args = MagicMock()
        args.input_ary_path = input_path
        args.label_ary_path = label_path

        from seq_run import SeqData

        seq_data = SeqData(args=args, train_ratio=0.8)
        assert not th.isnan(seq_data.train_input_seq).any()


# ---------------------------------------------------------------------------
# Sequence runner initialisation
# ---------------------------------------------------------------------------


class TestSeqRunConfig:
    """Tests for the ConfigData-compatible data path resolution."""

    def test_config_data_paths_are_strings(self):
        """ConfigData produces string paths under the given data_dir."""
        from seq_data import ConfigData  # type: ignore

        cfg = ConfigData(data_dir="/tmp/testdata")
        assert isinstance(cfg.input_ary_path, str)
        assert isinstance(cfg.label_ary_path, str)
        assert isinstance(cfg.predict_ary_path, str)
        assert "/tmp/testdata" in cfg.input_ary_path

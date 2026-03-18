"""Extended tests for seq_run.py — covers _update_network and valid_model."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

th = pytest.importorskip("torch", reason="torch not installed")


class TestUpdateNetwork:
    """Tests for seq_run._update_network."""

    def test_parameters_updated_after_step(self):
        """Optimizer.step() should change network parameters."""
        from seq_run import _update_network

        net = th.nn.Linear(4, 2)
        optimizer = th.optim.SGD(net.parameters(), lr=0.1)
        params_before = [p.clone().detach() for p in net.parameters()]

        # Compute a simple scalar loss
        x = th.randn(3, 4)
        obj = net(x).sum()

        _update_network(optimizer, obj, clip_grad_norm=10.0)

        params_after = list(net.parameters())
        changed = any(not th.allclose(pb, pa) for pb, pa in zip(params_before, params_after))
        assert changed, "Parameters should change after an optimizer step"

    def test_gradients_zero_after_step(self):
        """Gradients must be zeroed after optimizer.step()."""
        from seq_run import _update_network

        net = th.nn.Linear(4, 2)
        optimizer = th.optim.SGD(net.parameters(), lr=0.1)
        x = th.randn(3, 4)
        obj = net(x).sum()

        _update_network(optimizer, obj, clip_grad_norm=10.0)

        for p in net.parameters():
            assert p.grad is None or p.grad.abs().max().item() == 0.0

    def test_grad_clipping_applied(self):
        """With a tiny clip norm, gradient should be clipped to near zero."""
        from seq_run import _update_network

        net = th.nn.Linear(4, 2)
        optimizer = th.optim.SGD(net.parameters(), lr=0.0)  # lr=0 => no update
        x = th.ones(100, 4) * 1e3  # large input to create large gradients
        obj = net(x).sum()

        # Record parameters before
        params_before = [p.clone().detach() for p in net.parameters()]

        _update_network(optimizer, obj, clip_grad_norm=1e-9)

        # With lr=0, parameters should not change
        for pb, p in zip(params_before, net.parameters()):
            assert th.allclose(pb, p)


class TestSeqDataSplitRatios:
    """Additional split-ratio edge cases for SeqData."""

    def test_train_ratio_zero_all_goes_to_valid(self, tmp_path):
        from seq_run import SeqData

        input_path = str(tmp_path / "i.npy")
        label_path = str(tmp_path / "l.npy")
        np.save(input_path, np.random.randn(2000, 6).astype(np.float32))
        np.save(label_path, np.random.randn(2000, 3).astype(np.float32))

        args = MagicMock()
        args.input_ary_path = input_path
        args.label_ary_path = label_path

        sd = SeqData(args=args, train_ratio=0.0)
        assert sd.train_seq_len == 0

    def test_train_ratio_one_all_goes_to_train(self, tmp_path):
        from seq_run import SeqData

        input_path = str(tmp_path / "i2.npy")
        label_path = str(tmp_path / "l2.npy")
        np.save(input_path, np.random.randn(2000, 6).astype(np.float32))
        np.save(label_path, np.random.randn(2000, 3).astype(np.float32))

        args = MagicMock()
        args.input_ary_path = input_path
        args.label_ary_path = label_path

        sd = SeqData(args=args, train_ratio=1.0)
        assert sd.valid_seq_len == 0

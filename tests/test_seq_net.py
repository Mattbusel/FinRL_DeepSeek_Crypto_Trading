"""Tests for RnnRegNet and NnSeqBnMLP in seq_net.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from seq_net import NnSeqBnMLP, RnnRegNet


INP_DIM = 10
MID_DIM = 16
OUT_DIM = 4
NUM_LAYERS = 2


@pytest.fixture
def rnn_net():
    return RnnRegNet(
        inp_dim=INP_DIM,
        mid_dim=MID_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_LAYERS,
    )


class TestRnnRegNetOutputShape:
    def test_batch1_seq10_output_shape(self, rnn_net):
        seq_len, batch = 10, 1
        inp = torch.randn(seq_len, batch, INP_DIM)
        out, _ = rnn_net(inp)
        assert out.shape == (seq_len, batch, OUT_DIM)

    def test_batch32_seq10_output_shape(self, rnn_net):
        seq_len, batch = 10, 32
        inp = torch.randn(seq_len, batch, INP_DIM)
        out, _ = rnn_net(inp)
        assert out.shape == (seq_len, batch, OUT_DIM)

    def test_batch1_seq1_output_shape(self, rnn_net):
        inp = torch.randn(1, 1, INP_DIM)
        out, _ = rnn_net(inp)
        assert out.shape == (1, 1, OUT_DIM)

    def test_output_dtype_is_float32(self, rnn_net):
        inp = torch.randn(5, 2, INP_DIM)
        out, _ = rnn_net(inp)
        assert out.dtype == torch.float32

    def test_hidden_state_returned_as_tuple(self, rnn_net):
        inp = torch.randn(5, 2, INP_DIM)
        _, hid = rnn_net(inp)
        assert isinstance(hid, tuple)
        assert len(hid) == 2  # (hid1, hid2)

    def test_hidden_state_none_input_accepted(self, rnn_net):
        inp = torch.randn(5, 2, INP_DIM)
        out, hid = rnn_net(inp, hid=None)
        assert out is not None
        assert hid is not None

    def test_hidden_state_reuse(self, rnn_net):
        inp = torch.randn(5, 2, INP_DIM)
        _, hid = rnn_net(inp)
        out2, _ = rnn_net(inp, hid=hid)
        assert out2.shape == (5, 2, OUT_DIM)

    def test_output_is_finite(self, rnn_net):
        inp = torch.randn(8, 4, INP_DIM)
        out, _ = rnn_net(inp)
        assert torch.isfinite(out).all()

    def test_no_gradient_leak_in_eval_mode(self, rnn_net):
        rnn_net.eval()
        inp = torch.randn(5, 2, INP_DIM)
        with torch.no_grad():
            out, _ = rnn_net(inp)
        assert out.shape == (5, 2, OUT_DIM)

    def test_different_seq_lengths_accepted(self, rnn_net):
        for seq_len in [1, 8, 64, 256]:
            inp = torch.randn(seq_len, 2, INP_DIM)
            out, _ = rnn_net(inp)
            assert out.shape == (seq_len, 2, OUT_DIM)


class TestNnSeqBnMLP:
    def test_output_shape_3d(self):
        net = NnSeqBnMLP(dims=[8, 16, 4])
        inp = torch.randn(5, 3, 8)
        out = net(inp)
        assert out.shape == (5, 3, 4)

    def test_output_shape_batch1(self):
        net = NnSeqBnMLP(dims=[8, 16, 4])
        inp = torch.randn(10, 1, 8)
        out = net(inp)
        assert out.shape == (10, 1, 4)

    def test_layer_norm_variant(self):
        net = NnSeqBnMLP(dims=[8, 16, 4], if_layer_norm=True)
        inp = torch.randn(5, 3, 8)
        out = net(inp)
        assert out.shape == (5, 3, 4)

    def test_output_is_finite(self):
        net = NnSeqBnMLP(dims=[8, 16, 4])
        inp = torch.randn(4, 2, 8)
        out = net(inp)
        assert torch.isfinite(out).all()


class TestGetObjValue:
    def test_loss_shape_reduced_by_wup_dim(self, rnn_net):
        seq_len, batch = 20, 3
        out = torch.randn(seq_len, batch, OUT_DIM)
        lab = torch.randn(seq_len, batch, OUT_DIM)
        criterion = torch.nn.MSELoss(reduction="none")
        obj = RnnRegNet.get_obj_value(criterion, out, lab, wup_dim=5)
        assert obj.shape[0] == seq_len - 5

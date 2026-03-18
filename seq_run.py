"""RNN training and inference runner for the LARSA factor-mining pipeline.

This module trains a :class:`RnnRegNet` to regress Alpha101 + sentiment
features onto short-term price-movement labels, then serialises predictions
to a numpy array consumed by the trade simulator.

Typical usage::

    python seq_run.py -1   # CPU
    python seq_run.py 0    # GPU 0
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_

from seq_data import ConfigData, convert_btc_csv_to_btc_npy

log = logging.getLogger(__name__)

TEN = th.Tensor


class SeqData:
    """Loads, splits, and serves batched training/validation sequences.

    Args:
        args: A :class:`seq_data.ConfigData` (or compatible object) that
            provides ``input_ary_path`` and ``label_ary_path`` attributes.
        train_ratio: Fraction of the sequence used for training.
            When set to ``0.0`` or ``1.0`` the entire sequence is assigned to
            the respective split.

    Raises:
        FileNotFoundError: If either array file is missing and
            :func:`seq_data.convert_btc_csv_to_btc_npy` does not create them.
        AssertionError: If NaN values remain in the input array after
            sanitisation.
    """

    def __init__(self, args: ConfigData, train_ratio: float = 0.8) -> None:
        input_ary_path = args.input_ary_path
        label_ary_path = args.label_ary_path

        if not os.path.exists(label_ary_path) or not os.path.exists(input_ary_path):
            log.info(
                "seq_data.generate -- array files missing, running convert_btc_csv_to_btc_npy"
            )
            convert_btc_csv_to_btc_npy(args=args)

        input_seq = np.load(input_ary_path)
        input_seq = np.nan_to_num(input_seq, nan=0.0, neginf=0.0, posinf=0.0)
        input_seq = th.tensor(input_seq, dtype=th.float32)
        assert th.isnan(input_seq).sum() == 0

        label_seq = np.load(label_ary_path)
        label_seq = th.tensor(label_seq, dtype=th.float32)
        assert th.isnan(label_seq).sum() == 0

        seq_len = label_seq.shape[0]
        self.input_dim: int = input_seq.shape[1]
        self.label_dim: int = label_seq.shape[1]

        seq_i0 = 0
        seq_i1 = int(seq_len * train_ratio)
        seq_i2 = int(seq_len - 1024) if 0 < train_ratio < 1 else seq_len
        self.train_seq_len: int = seq_i1 - seq_i0
        self.valid_seq_len: int = seq_i2 - seq_i1

        self.train_input_seq: TEN = input_seq[seq_i0:seq_i1]
        self.valid_input_seq: TEN = input_seq[seq_i1:seq_i2]
        self.train_label_seq: TEN = label_seq[seq_i0:seq_i1]
        self.valid_label_seq: TEN = label_seq[seq_i1:seq_i2]

    def sample_for_train(
        self,
        batch_size: int = 32,
        seq_len: int = 4096,
        device: th.device = th.device("cpu"),
    ) -> Tuple[TEN, TEN]:
        """Draw a random batch of fixed-length sub-sequences from the training split.

        Args:
            batch_size: Number of independent sub-sequences to return.
            seq_len: Length of each sub-sequence in time steps.
            device: PyTorch device to place the returned tensors on.

        Returns:
            A tuple ``(inputs, labels)`` each of shape
            ``(seq_len, batch_size, feature_dim)``.
        """
        i0s = np.random.randint(1024, self.train_seq_len - seq_len, size=batch_size)
        ids = np.arange(seq_len)[None, :].repeat(batch_size, axis=0) + i0s[:, None]

        input_ary = self.train_input_seq[ids, :].permute(1, 0, 2).to(device)
        label_seq = self.train_label_seq[ids, :].permute(1, 0, 2).to(device)
        return input_ary, label_seq


def _update_network(
    optimizer: th.optim.Optimizer,
    obj: TEN,
    clip_grad_norm: float,
) -> None:
    """Back-propagate *obj* and step *optimizer* with gradient clipping.

    Args:
        optimizer: AdamW or similar optimiser to step.
        obj: Scalar loss tensor.
        clip_grad_norm: Maximum gradient norm threshold.
    """
    optimizer.zero_grad()
    obj.backward()
    clip_grad_norm_(
        parameters=optimizer.param_groups[0]["params"],
        max_norm=clip_grad_norm,
    )
    optimizer.step()


def train_model(gpu_id: int) -> None:
    """Train the RNN regression model and save weights and predictions.

    Loads (or generates) the pre-processed numpy arrays, trains
    :class:`seq_net.RnnRegNet`, applies early stopping, saves the best
    checkpoint, and writes a full-validation prediction array to disk.

    Args:
        gpu_id: CUDA device index.  Pass ``-1`` to force CPU.
    """
    device = th.device(
        f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
    )

    batch_size = 256
    mid_dim = 128
    num_layers = 4
    epoch = 2 ** 8
    wup_dim = 64
    valid_gap = 128
    num_patience = 8
    weight_decay = 1e-4
    learning_rate = 1e-3
    clip_grad_norm = 2.0

    out_dir = "./output"

    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.6)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    from seq_net import RnnRegNet  # noqa: PLC0415

    net = RnnRegNet(
        inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers
    ).to(device)
    optimizer = th.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = th.nn.MSELoss(reduction="none")

    from seq_record import Evaluator  # noqa: PLC0415

    evaluator = Evaluator(out_dir=out_dir)

    seq_len = 2 ** 8
    train_times = int(seq_data.train_seq_len / seq_len / batch_size * epoch)
    log.info(
        "train_model.start",
        extra={"train_seq_len": seq_data.train_seq_len, "train_times": train_times},
    )

    for step_idx in range(train_times):
        th.set_grad_enabled(True)
        net.train()
        inp, lab = seq_data.sample_for_train(
            batch_size=batch_size, seq_len=seq_len, device=device
        )
        out, _ = net(inp)
        obj = net.get_obj_value(criterion=criterion, out=out, lab=lab, wup_dim=wup_dim)
        _update_network(optimizer, obj.mean(), clip_grad_norm)

        evaluator.update_obj_train(obj=obj)

        if (step_idx % valid_gap == 0) or (step_idx == train_times - 1):
            th.set_grad_enabled(False)
            evaluator.update_obj_train(obj=None)

            net.eval()
            for _ in range(int(seq_data.valid_seq_len / seq_len / batch_size)):
                inp, lab = seq_data.sample_for_train(
                    batch_size=batch_size, seq_len=seq_len, device=device
                )
                out, _ = net(inp)

                valid_out_len = min(out.shape[0], lab.shape[0])
                out = out[wup_dim:valid_out_len, :, :]
                lab = lab[wup_dim:valid_out_len, :, :]
                obj = criterion(out, lab)
                evaluator.update_obj_valid(obj=obj)
            del inp, lab, out

            evaluator.update_obj_valid(obj=None)
            evaluator.log_print(step_idx=step_idx)
            evaluator.draw_train_valid_loss_curve(gpu_id=gpu_id)

            if evaluator.patience > num_patience:
                log.info("train_model.early_stop", extra={"step": step_idx})
                break
            if evaluator.patience == 0:
                best_valid_loss = evaluator.best_valid_loss
                ckpt_path = f"{out_dir}/net_{step_idx:06}_{best_valid_loss:06.3f}.pth"
                th.save(net.state_dict(), ckpt_path)
                log.info("train_model.checkpoint_saved", extra={"path": ckpt_path})

    predict_net_path = args.predict_net_path
    th.save(net.state_dict(), predict_net_path)
    log.info("train_model.network_saved", extra={"path": predict_net_path})

    predict_ary = np.empty_like(seq_data.valid_label_seq)
    hid: Optional[TEN] = None

    log.info(
        "train_model.inference_start",
        extra={
            "valid_seq_len": seq_data.valid_seq_len,
            "valid_times": seq_data.valid_seq_len // seq_len,
        },
    )
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)

    predict_ary_path = args.predict_ary_path
    np.save(predict_ary_path, predict_ary)
    log.info("train_model.predictions_saved", extra={"path": predict_ary_path})


def valid_model(gpu_id: int) -> None:
    """Run inference using a saved RNN checkpoint and save predictions to disk.

    Loads the checkpoint at :attr:`seq_data.ConfigData.predict_net_path`,
    runs inference over the full (train_ratio=0) split, and saves the
    prediction array.

    Args:
        gpu_id: CUDA device index.  Pass ``-1`` to force CPU.
    """
    device = th.device(
        f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
    )
    th.set_grad_enabled(False)

    mid_dim = 128
    num_layers = 4

    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.0)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    predict_net_path = args.predict_net_path
    predict_ary_path = args.predict_ary_path

    from seq_net import RnnRegNet  # noqa: PLC0415

    net = RnnRegNet(
        inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers
    ).to(device)
    net.load_state_dict(
        th.load(predict_net_path, map_location=lambda storage, loc: storage)
    )

    predict_ary = np.empty_like(seq_data.valid_label_seq)
    hid: Optional[TEN] = None

    seq_len = 2 ** 9
    log.info(
        "valid_model.start",
        extra={
            "valid_seq_len": seq_data.valid_seq_len,
            "valid_times": seq_data.valid_seq_len // seq_len,
        },
    )
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)

    np.save(predict_ary_path, predict_ary)
    log.info("valid_model.predictions_saved", extra={"path": predict_ary_path})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    convert_btc_csv_to_btc_npy()
    train_model(gpu_id=GPU_ID)
    valid_model(gpu_id=GPU_ID)

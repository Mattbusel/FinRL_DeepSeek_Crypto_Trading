"""Training evaluation recorder and loss-curve plotter for the LARSA RNN pipeline.

Provides :class:`Evaluator`, which tracks per-step training and validation
loss tensors, implements early stopping via a patience counter, and renders
a three-panel loss-curve figure to disk.

Example::

    from seq_record import Evaluator
    evaluator = Evaluator(out_dir="./output")
    evaluator.update_obj_train(obj=loss_tensor)
    evaluator.update_obj_train(obj=None)   # finalise epoch average
    evaluator.log_print(step_idx=0)
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import torch as th

log = logging.getLogger(__name__)

TEN = th.Tensor


def import_matplotlib_in_server():
    """Import matplotlib with the non-interactive ``Agg`` backend.

    This allows figure generation on headless servers without an X display.

    Returns:
        The :mod:`matplotlib.pyplot` module configured for off-screen rendering.
    """
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def skip_method_if_report_disabled(method):
    """Decorator that skips a method when ``self.if_report`` is ``False``.

    Args:
        method: The bound method to wrap.

    Returns:
        A wrapper that returns ``None`` early when reporting is disabled.
    """

    def wrapper(self, *args, **kwargs):
        if not self.if_report:
            return None
        return method(self, *args, **kwargs)

    return wrapper


class Evaluator:
    """Tracks training/validation loss history and manages early stopping.

    Args:
        out_dir: Directory where checkpoint figures and CSV logs are saved.
            Created automatically if absent.

    Attributes:
        patience: Number of consecutive validation steps without improvement.
        best_valid_loss: Lowest validation loss seen so far.
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir: str = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.step_idx: list[int] = []
        self.step_sec: list[int] = []

        self.tmp_train: list[TEN] = []
        self.obj_train: list[TEN] = []

        self.tmp_valid: list[TEN] = []
        self.obj_valid: list[TEN] = []

        self.start_time: float = time.time()

        self.patience: int = 0
        self.best_valid_loss: float = th.inf

    def update_obj_train(self, obj: TEN | None = None) -> None:
        """Accumulate a training loss batch or finalise the epoch average.

        Call with a loss tensor to append to the running buffer; call with
        ``None`` to flush the buffer into :attr:`obj_train`.

        Args:
            obj: Per-element loss tensor, or ``None`` to compute the epoch
                mean and clear the buffer.
        """
        if obj is None:
            obj_avg = th.stack(self.tmp_train).mean(dim=0)
            self.tmp_train.clear()
            self.obj_train.append(obj_avg)
        else:
            self.tmp_train.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_obj_valid(self, obj: TEN | None = None) -> None:
        """Accumulate a validation loss batch or finalise the epoch average.

        Args:
            obj: Per-element loss tensor, or ``None`` to flush the buffer.
        """
        if obj is None:
            obj_avg = th.stack(self.tmp_valid).mean(dim=0)
            self.tmp_valid.clear()
            self.obj_valid.append(obj_avg)
        else:
            self.tmp_valid.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_patience_and_best_valid_loss(self) -> None:
        """Update :attr:`patience` and :attr:`best_valid_loss` from the latest epoch.

        Resets *patience* to zero on improvement; increments it otherwise.
        """
        valid_loss = self.obj_valid[-1].mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = float(valid_loss)
            self.patience = 0
        else:
            self.patience += 1

    def log_print(self, step_idx: int) -> None:
        """Log training/validation statistics for the current step.

        Also updates :attr:`patience` and :attr:`best_valid_loss`.

        Args:
            step_idx: Current global training step index.
        """
        self.step_idx.append(step_idx)
        self.step_sec.append(int(time.time()))

        avg_train = self.obj_train[-1].numpy()
        avg_valid = self.obj_valid[-1].numpy()
        time_used = int(time.time() - self.start_time)

        valid_loss = float(self.obj_valid[-1].mean())
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience = 0
        else:
            self.patience += 1

        avg_valid_percent = (avg_valid * 1000).astype(int)
        log.info(
            "train.step %6d  %6d sec  patience %d | train %9.3e  valid %9.3e  pct%s",
            step_idx,
            time_used,
            self.patience,
            float(avg_train.mean()),
            float(avg_valid.mean()),
            avg_valid_percent,
        )

    def draw_train_valid_loss_curve(
        self,
        gpu_id: int = -1,
        figure_path: str = "",
        ignore_ratio: float = 0.05,
    ) -> None:
        """Save a three-panel training/validation loss figure to disk.

        The three panels show: (1) average train vs. valid loss, (2) per-label
        train loss, and (3) per-label validation loss.

        Args:
            gpu_id: GPU identifier used in the output filename when
                *figure_path* is not provided.
            figure_path: Explicit destination path for the figure.  Defaults
                to ``{out_dir}/a_figure_loss_curve_{gpu_id}.jpg``.
            ignore_ratio: Fraction of early steps to omit from the plot to
                reduce visual noise.
        """
        figure_path = (
            figure_path if figure_path else f"{self.out_dir}/a_figure_loss_curve_{gpu_id}.jpg"
        )
        step_idx: list = self.step_idx
        step_sec: list = self.step_sec

        curve_num = len(step_idx)
        if curve_num < 2:
            return

        ignore_num = int(curve_num * ignore_ratio)
        step_idx = step_idx[ignore_num:]
        step_sec = step_sec[ignore_num:]
        assert len(step_idx) == len(step_sec)

        obj_train = th.stack(self.obj_train[ignore_num:], dim=0).detach().cpu().numpy()
        avg_train = obj_train.mean(axis=1)
        obj_valid = th.stack(self.obj_valid[ignore_num:], dim=0).detach().cpu().numpy()
        avg_valid = obj_valid.mean(axis=1)

        plt = import_matplotlib_in_server()

        fig, axs = plt.subplots(3)
        fig.set_size_inches(12, 20)
        fig.suptitle("Loss Curve", y=0.98)
        alpha = 0.8
        tl_color, tl_style, tl_width = "black", "-", 3
        vl_color, vl_style, vl_width = "black", "-.", 3

        xs = step_idx

        res_df = pd.DataFrame(
            np.array([avg_train, avg_valid]).T,
            index=xs,
            columns=["train_avg_loss", "valid_avg_loss"],
        )
        res_df.to_csv(os.path.join(os.path.dirname(figure_path), "loss_df.csv"))

        ax0 = axs[0]
        ax0.plot(xs, avg_train, color=tl_color, linestyle=tl_style, label="TrainAvg")
        ax0.plot(xs, avg_valid, color=vl_color, linestyle=vl_style, label="ValidAvg")
        ax0.legend()
        ax0.grid()

        ax1 = axs[1]
        ax1.plot(
            xs,
            avg_train,
            color=tl_color,
            linestyle=tl_style,
            linewidth=tl_width,
            label="TrainAvg",
        )
        for label_i in range(obj_train.shape[1]):
            ax1.plot(xs, obj_train[:, label_i], alpha=alpha, label=f"Lab-{label_i}")
        ax1.legend()
        ax1.grid()

        ax2 = axs[2]
        ax2.plot(
            xs,
            avg_valid,
            color=vl_color,
            linestyle=vl_style,
            linewidth=vl_width,
            label="ValidAvg",
        )
        for label_i in range(obj_valid.shape[1]):
            ax2.plot(xs, obj_valid[:, label_i], alpha=alpha, label=f"Lab-{label_i}")
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.05, wspace=0.05)
        plt.savefig(figure_path, dpi=200)
        plt.close("all")

    def close(self, gpu_id: int) -> None:
        """Log final GPU memory usage and save the loss curve.

        Args:
            gpu_id: CUDA device index used to query memory statistics.
        """
        device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu")

        if th.cuda.is_available():
            max_memo = th.cuda.max_memory_allocated(device=device)
            dev_memo = th.cuda.get_device_properties(gpu_id).total_memory
        else:
            max_memo = 0.0
            dev_memo = 1.0  # avoid division by zero

        self.draw_train_valid_loss_curve(gpu_id=gpu_id)
        log.info(
            "evaluator.close GPU_GB=%.2f GPU_ratio=%.2f TimeUsed=%d",
            max_memo / 2**30,
            max_memo / dev_memo,
            int(time.time() - self.start_time),
        )

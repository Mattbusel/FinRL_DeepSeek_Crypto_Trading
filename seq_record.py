import os
import time
import numpy as np
import torch as th
import pandas as pd

TEN = th.Tensor


def import_matplotlib_in_server():
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    return plt


def skip_method_if_report_disabled(method):
    def wrapper(self, *args, **kwargs):
        if not self.if_report:
            return None
        return method(self, *args, **kwargs)

    return wrapper


class Evaluator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

        '''Training Log'''
        self.step_idx = []
        self.step_sec = []

        self.tmp_train = []
        self.obj_train = []

        self.tmp_valid = []
        self.obj_valid = []

        '''Timing module'''
        self.start_time = time.time()

        '''Automatically stop training components'''
        self.patience = 0
        self.best_valid_loss = th.inf

    def update_obj_train(self, obj=None):
        if obj is None:
            obj_avg = th.stack(self.tmp_train).mean(dim=0)
            self.tmp_train[:] = []
            self.obj_train.append(obj_avg)
        else:
            self.tmp_train.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_obj_valid(self, obj=None):
        if obj is None:
            obj_avg = th.stack(self.tmp_valid).mean(dim=0)
            self.tmp_valid[:] = []
            self.obj_valid.append(obj_avg)
        else:
            self.tmp_valid.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_patience_and_best_valid_loss(self):
        valid_loss = self.obj_valid[-1].mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience = 0
        else:
            self.patience += 1

    def log_print(self, step_idx: int):
        self.step_idx.append(step_idx)
        self.step_sec.append(int(time.time()))

        avg_train = self.obj_train[-1].numpy()
        avg_valid = self.obj_valid[-1].numpy()
        time_used = int(time.time() - self.start_time)

        '''update_patience_and_best_valid_loss'''
        valid_loss = self.obj_valid[-1].mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience = 0
        else:
            self.patience += 1

        avg_valid_percent = (avg_valid * 1000).astype(int)
        print(f"{step_idx:>6}  {time_used:>6} sec  patience {self.patience:<4}"
              f"| train {avg_train.mean():9.3e}  valid {avg_valid.mean():9.3e}  %0{avg_valid_percent}")

    def draw_train_valid_loss_curve(self, gpu_id: int = -1, figure_path='', ignore_ratio: float = 0.05):
        figure_path = figure_path if figure_path else f"{self.out_dir}/a_figure_loss_curve_{gpu_id}.jpg"
        step_idx: list = self.step_idx  # write behind `self.log_print` update step_idx
        step_sec: list = self.step_sec  # write behind `self.log_print` update step_sec

        curve_num = len(step_idx)
        if curve_num < 2:
            return

        '''ignore_ratio'''
        ignore_num = int(curve_num * ignore_ratio)
        step_idx = step_idx[ignore_num:]
        step_sec = step_sec[ignore_num:]
        assert len(step_idx) == len(step_sec)

        '''ignore_ratio before mean'''
        obj_train = th.stack(self.obj_train[ignore_num:], dim=0).detach().cpu().numpy()
        avg_train = obj_train.mean(axis=1)
        obj_valid = th.stack(self.obj_valid[ignore_num:], dim=0).detach().cpu().numpy()
        avg_valid = obj_valid.mean(axis=1)

        '''plot subplots'''
        plt = import_matplotlib_in_server()

        fig, axs = plt.subplots(3)
        fig.set_size_inches(12, 20)
        fig.suptitle('Loss Curve', y=0.98)
        alpha = 0.8
        tl_color, tl_style, tl_width = 'black', '-', 3  # train line
        vl_color, vl_style, vl_width = 'black', '-.', 3  # valid line

        xs = step_idx
        ys_train = avg_train
        ys_valid = avg_valid

        # Make a table and save it
        res_df = pd.DataFrame(np.array([ys_train, ys_valid]).T, index=xs, columns=['train_avg_loss', 'valid_avg_loss'])
        res_df.to_csv(os.path.join(os.path.dirname(figure_path), 'loss_df.csv'))

        ax0 = axs[0]
        ax0.plot(xs, ys_train, color=tl_color, linestyle=tl_style, label='TrainAvg')
        ax0.plot(xs, ys_valid, color=vl_color, linestyle=vl_style, label='ValidAvg')
        ax0.legend()
        ax0.grid()

        ax1 = axs[1]
        ax1.plot(xs, ys_train, color=tl_color, linestyle=tl_style, linewidth=tl_width, label='TrainAvg')
        for label_i in range(obj_train.shape[1]):
            ax1.plot(xs, obj_train[:, label_i], alpha=alpha, label=f'Lab-{label_i}')
        ax1.legend()
        ax1.grid()

        ax2 = axs[2]
        ax2.plot(xs, ys_valid, color=vl_color, linestyle=vl_style, linewidth=vl_width, label='ValidAvg')
        for label_i in range(obj_valid.shape[1]):
            ax2.plot(xs, obj_valid[:, label_i], alpha=alpha, label=f'Lab-{label_i}')
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.05, wspace=0.05)
        plt.savefig(figure_path, dpi=200)
        plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
        # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()

    def close(self, gpu_id: int):
        device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        if th.cuda.is_available():
            max_memo = th.cuda.max_memory_allocated(device=device)
            dev_memo = th.cuda.get_device_properties(gpu_id).total_memory
        else:
            max_memo = 0.0
            dev_memo = 0.0

        self.draw_train_valid_loss_curve(gpu_id=gpu_id)
        print(f"GPU(GB)    {max_memo / 2 ** 30:.2f}    "
              f"GPU(ratio) {max_memo / dev_memo:.2f}    "
              f"TimeUsed {int(time.time() - self.start_time)}")
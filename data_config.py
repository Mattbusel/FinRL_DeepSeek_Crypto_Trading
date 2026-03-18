"""Data path configuration for the LARSA trading system.

Provides a lightweight container for resolved file-system paths so that
every module can locate training and prediction artifacts from a single
source of truth.

Example::

    from data_config import ConfigData
    cfg = ConfigData(data_dir="./data")
    print(cfg.csv_path)
"""

from __future__ import annotations


class ConfigData:
    """Resolved file-system paths for training and inference artifacts.

    Args:
        data_dir: Root directory that contains all data files.
            Defaults to ``"./data"``.

    Raises:
        TypeError: If *data_dir* is not a string.
    """

    def __init__(self, data_dir: str = "./data") -> None:
        if not isinstance(data_dir, str):
            raise TypeError(
                f"data_dir must be a str, got {type(data_dir).__name__!r}"
            )
        self.data_dir: str = data_dir
        self.csv_path: str = f"{data_dir}/BTC_1sec_with_sentiment_risk_train.csv"

        #: NumPy array of RNN input features (Alpha101 + news signals).
        self.input_ary_path: str = f"{data_dir}/BTC_1sec_input.npy"

        #: NumPy array of RNN regression targets (future price signals).
        self.label_ary_path: str = f"{data_dir}/BTC_1sec_label.npy"

        #: Saved RNN network weights used for factor prediction.
        self.predict_net_path: str = f"{data_dir}/BTC_1sec_predict_net.pth"
        self.predict_ary_path: str = f"{data_dir}/BTC_1sec_predict.npy"
"""Build the train/test data the trading environment reads.

Inputs are the two public files from the FinRL Contest 2024 Task 1 starter
kit (Google Drive folder ``1Okd8fyB7n93N1Z5HEnlpb-q8x5FfSF1Z``):

* ``BTC_1sec.csv``          1-second BTC limit-order-book snapshots
* ``BTC_1sec_predict.npy``  8 RNN factors per second, computed by the
                            contest organisers on the same CSV

Optionally, a news file scored by ``deepseek_signals.py`` can be merged in.
Each second then carries the most recent ``sentiment_score`` and
``risk_score`` (1 to 5).  Seconds with no scored news before them get the
neutral value 3, which is also what every row gets when no news file is given.

Outputs (the column subset ``trade_simulator.py`` needs, plus the factors)::

    data/train/BTC_1sec_with_sentiment_risk_train.csv
    data/train/BTC_1sec_predict.npy
    data/test/BTC_1sec_with_sentiment_risk_train.csv
    data/test/BTC_1sec_predict.npy

Train on ``data/train`` and evaluate on ``data/test`` by pointing the
``LARSA_DATA_DIR`` environment variable at one or the other.

Usage::

    python scripts/prepare_data.py --download            # fetch + split
    python scripts/prepare_data.py --source path/to/dir  # already downloaded
    python scripts/prepare_data.py --signals data/news_with_signals.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Okd8fyB7n93N1Z5HEnlpb-q8x5FfSF1Z"
PRICE_COLUMNS = ["system_time", "midpoint", "spread", "bids_distance_3", "asks_distance_3"]
TIME_COLUMNS = ("system_time", "timestamp", "datetime", "date", "published_at", "time")
NEUTRAL = 3.0
CSV_NAME = "BTC_1sec_with_sentiment_risk_train.csv"
NPY_NAME = "BTC_1sec_predict.npy"


def download(dest: str) -> str:
    """Download the contest data folder with gdown and return the data dir."""
    try:
        import gdown  # type: ignore[import]
    except ImportError:
        sys.exit("gdown is needed for --download:  pip install gdown")
    os.makedirs(dest, exist_ok=True)
    gdown.download_folder(DRIVE_FOLDER, output=dest, quiet=False)
    for root, _dirs, files in os.walk(dest):
        if "BTC_1sec.csv" in files and NPY_NAME in files:
            return root
    sys.exit(f"Download finished but BTC_1sec.csv / {NPY_NAME} were not found under {dest}")


def attach_signals(df: pd.DataFrame, signals_path: str | None) -> pd.DataFrame:
    """Add sentiment_score / risk_score columns, forward-filled per second."""
    df["sentiment_score"] = NEUTRAL
    df["risk_score"] = NEUTRAL
    if not signals_path:
        return df

    news = pd.read_csv(signals_path)
    time_col = next((c for c in TIME_COLUMNS if c in news.columns), None)
    if time_col is None:
        sys.exit(f"{signals_path} needs one of these time columns: {', '.join(TIME_COLUMNS)}")
    missing = {"sentiment_score", "risk_score"} - set(news.columns)
    if missing:
        sys.exit(f"{signals_path} is missing {sorted(missing)}; run deepseek_signals.py first")

    news = news[[time_col, "sentiment_score", "risk_score"]].dropna()
    news["_t"] = pd.to_datetime(news[time_col], utc=True)
    news = news.sort_values("_t")

    left = pd.DataFrame({"_t": pd.to_datetime(df["system_time"], utc=True)})
    merged = pd.merge_asof(left, news[["_t", "sentiment_score", "risk_score"]], on="_t")
    df["sentiment_score"] = merged["sentiment_score"].fillna(NEUTRAL).to_numpy()
    df["risk_score"] = merged["risk_score"].fillna(NEUTRAL).to_numpy()
    print(f"| merged {len(news)} scored news items from {signals_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="data/raw", help="dir with BTC_1sec.csv + BTC_1sec_predict.npy")
    parser.add_argument("--download", action="store_true", help="fetch the contest data first (needs gdown)")
    parser.add_argument("--signals", default=None, help="news CSV scored by deepseek_signals.py")
    parser.add_argument("--out", default="data", help="output root (train/ and test/ are created)")
    parser.add_argument("--test-frac", type=float, default=0.3, help="held-out tail fraction (default 0.3)")
    args = parser.parse_args()

    source = download(args.source) if args.download else args.source
    csv_path = f"{source}/BTC_1sec.csv"
    npy_path = f"{source}/{NPY_NAME}"
    for path in (csv_path, npy_path):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\nrun with --download, or fetch it from {DRIVE_FOLDER}")

    print(f"| reading {csv_path}")
    df = pd.read_csv(csv_path, usecols=PRICE_COLUMNS)
    factors = np.load(npy_path)

    # The factor array is shorter than the CSV (RNN warm-up); align on the tail
    # exactly like TradeSimulator does.
    df = df.iloc[-factors.shape[0]:].reset_index(drop=True)
    df = attach_signals(df, args.signals)

    n = len(df)
    cut = int(n * (1.0 - args.test_frac))
    for name, sl in (("train", slice(0, cut)), ("test", slice(cut, n))):
        out_dir = f"{args.out}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        part = df.iloc[sl]
        part.to_csv(f"{out_dir}/{CSV_NAME}", index=False)
        np.save(f"{out_dir}/{NPY_NAME}", factors[sl].astype(np.float32))
        t0, t1 = part["system_time"].iloc[0][:19], part["system_time"].iloc[-1][:19]
        print(f"| {name:5}  {len(part):>7,} seconds  {t0} .. {t1}  -> {out_dir}")


if __name__ == "__main__":
    main()

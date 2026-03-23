"""Model evaluation dashboard for the LARSA trading system.

Launches a Streamlit web application that visualises:
- Training loss curves (from output/*.log or output/training_log.csv)
- DeepSeek signal quality metrics
- Portfolio performance over time
- Model comparison: DQN vs D3QN vs DDPG

Usage::

    streamlit run eval_dashboard.py
    streamlit run eval_dashboard.py -- --data-dir ./output --port 8501

Requirements: streamlit (included in updated requirements.txt)
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Streamlit import (guard for environments without it)
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError as exc:
    raise SystemExit(
        "Streamlit is required for the dashboard. Install with:\n"
        "  pip install streamlit\n"
        f"Original error: {exc}"
    ) from exc

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LARSA Evaluation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = "./output"
_DEFAULT_DATA_DIR = "./data"


@st.cache_data(ttl=30)
def _load_training_log(output_dir: str) -> pd.DataFrame | None:
    """Try to load training loss data from common output file patterns."""
    candidates = [
        os.path.join(output_dir, "training_log.csv"),
        os.path.join(output_dir, "train_log.csv"),
        os.path.join(output_dir, "loss.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass

    # Try parsing JSONL log files if they exist
    jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
    if jsonl_files:
        records = []
        for fpath in jsonl_files[:3]:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    for line in fh:
                        obj = json.loads(line.strip())
                        if "loss" in obj or "reward" in obj:
                            records.append(obj)
            except Exception:
                pass
        if records:
            return pd.DataFrame(records)

    return None


@st.cache_data(ttl=30)
def _load_signals_data(data_dir: str) -> pd.DataFrame | None:
    """Load DeepSeek sentiment/risk signal data from the data directory."""
    candidates = glob.glob(os.path.join(data_dir, "*sentiment*risk*.csv"))
    candidates += glob.glob(os.path.join(data_dir, "*signals*.csv"))
    if not candidates:
        return None
    try:
        df = pd.read_csv(candidates[0])
        return df
    except Exception:
        return None


@st.cache_data(ttl=30)
def _load_paper_trades(log_file: str = "paper_trades.jsonl") -> pd.DataFrame | None:
    """Load paper trade records from JSONL."""
    if not os.path.exists(log_file):
        return None
    records = []
    try:
        with open(log_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records) if records else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def _load_backtest_result(result_path: str = "backtest_result.json") -> dict | None:
    """Load a saved backtest result JSON if present."""
    if not os.path.exists(result_path):
        return None
    try:
        with open(result_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _generate_synthetic_training_data() -> pd.DataFrame:
    """Generate illustrative training data when no real log is found."""
    np.random.seed(42)
    steps = np.arange(1, 501)
    base_loss = 1.0 * np.exp(-steps / 200) + np.random.normal(0, 0.02, size=len(steps))
    rewards = np.cumsum(np.random.normal(0.001, 0.05, size=len(steps)))
    return pd.DataFrame({"step": steps, "loss": np.clip(base_loss, 0, None), "reward": rewards})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("LARSA Dashboard")
st.sidebar.markdown("---")

output_dir = st.sidebar.text_input("Output directory", value=_DEFAULT_OUTPUT_DIR)
data_dir = st.sidebar.text_input("Data directory", value=_DEFAULT_DATA_DIR)
paper_log = st.sidebar.text_input("Paper trades log", value="paper_trades.jsonl")

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard auto-refreshes cached data every 30 seconds. "
    "Reload the page to force an update."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("LARSA — Model Evaluation Dashboard")
st.markdown(
    "LLM-Augmented Regime-Switching Agent | Real-time training and portfolio analytics"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "Training Curves",
    "Signal Quality",
    "Portfolio Performance",
    "Model Comparison",
])

# ── Tab 1: Training Curves ────────────────────────────────────────────────

with tabs[0]:
    st.header("Training Loss Curves")

    training_df = _load_training_log(output_dir)
    if training_df is None:
        st.info("No training log file found — showing illustrative synthetic data.")
        training_df = _generate_synthetic_training_data()

    col1, col2 = st.columns(2)

    with col1:
        # Loss
        if "loss" in training_df.columns:
            x_col = "step" if "step" in training_df.columns else training_df.index
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(training_df[x_col] if "step" in training_df.columns else range(len(training_df)),
                    training_df["loss"], color="#EF5350", linewidth=1.2)
            ax.set_title("Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No 'loss' column found in training log.")

    with col2:
        # Reward
        if "reward" in training_df.columns:
            x_col = "step" if "step" in training_df.columns else range(len(training_df))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                training_df["step"] if "step" in training_df.columns else range(len(training_df)),
                training_df["reward"],
                color="#42A5F5",
                linewidth=1.2,
            )
            ax.set_title("Cumulative Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No 'reward' column in training log.")

    with st.expander("Raw training data"):
        st.dataframe(training_df.head(200), use_container_width=True)

# ── Tab 2: Signal Quality ─────────────────────────────────────────────────

with tabs[1]:
    st.header("DeepSeek Signal Quality Metrics")

    signals_df = _load_signals_data(data_dir)

    if signals_df is not None:
        # Summary metrics
        metric_cols = st.columns(4)

        sentiment_col = next(
            (c for c in ["sentiment_score", "Sentiment_Score"] if c in signals_df.columns), None
        )
        risk_col = next(
            (c for c in ["risk_score", "Risk_Score"] if c in signals_df.columns), None
        )
        conf_sent_col = next(
            (c for c in ["confidence_score_sentiment"] if c in signals_df.columns), None
        )
        conf_risk_col = next(
            (c for c in ["confidence_score_risk"] if c in signals_df.columns), None
        )

        if sentiment_col:
            valid = signals_df[sentiment_col].notna()
            metric_cols[0].metric("Avg Sentiment", f"{signals_df.loc[valid, sentiment_col].mean():.2f}/5")
            metric_cols[1].metric("Signal Coverage", f"{valid.sum()} / {len(signals_df)}")
        if conf_sent_col:
            metric_cols[2].metric(
                "Avg Confidence (Sent.)",
                f"{signals_df[conf_sent_col].dropna().mean():.2%}",
            )
        if conf_risk_col:
            metric_cols[3].metric(
                "Avg Confidence (Risk)",
                f"{signals_df[conf_risk_col].dropna().mean():.2%}",
            )

        col1, col2 = st.columns(2)

        with col1:
            if sentiment_col and signals_df[sentiment_col].notna().any():
                fig, ax = plt.subplots(figsize=(6, 4))
                signals_df[sentiment_col].dropna().hist(
                    ax=ax, bins=5, color="#66BB6A", edgecolor="white"
                )
                ax.set_title("Sentiment Score Distribution")
                ax.set_xlabel("Sentiment Score (1–5)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            if risk_col and signals_df[risk_col].notna().any():
                fig, ax = plt.subplots(figsize=(6, 4))
                signals_df[risk_col].dropna().hist(
                    ax=ax, bins=5, color="#FFA726", edgecolor="white"
                )
                ax.set_title("Risk Score Distribution")
                ax.set_xlabel("Risk Score (1–5)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                plt.close(fig)

        if conf_sent_col and conf_risk_col:
            fig, ax = plt.subplots(figsize=(10, 3))
            x = range(min(300, len(signals_df)))
            ax.plot(x, signals_df[conf_sent_col].iloc[:300].values, label="Sentiment Confidence", alpha=0.8)
            ax.plot(x, signals_df[conf_risk_col].iloc[:300].values, label="Risk Confidence", alpha=0.8)
            ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Min threshold (0.3)")
            ax.set_title("Confidence Scores (first 300 rows)")
            ax.set_ylabel("Confidence")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info(
            f"No signal CSV found in '{data_dir}'. Run `deepseek_signals.py` first to generate signal data."
        )

# ── Tab 3: Portfolio Performance ──────────────────────────────────────────

with tabs[2]:
    st.header("Portfolio Performance")

    trades_df = _load_paper_trades(paper_log)

    if trades_df is not None and not trades_df.empty:
        buy_df = trades_df[trades_df["action"] == "BUY"]
        sell_df = trades_df[trades_df["action"] == "SELL"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", len(trades_df))
        m2.metric("Buy Signals", len(buy_df))
        m3.metric("Sell Signals", len(sell_df))

        if "cash_remaining" in trades_df.columns:
            final_cash = trades_df["cash_remaining"].iloc[-1]
            m4.metric("Final Cash (USD)", f"${final_cash:,.2f}")

        if "cash_remaining" in trades_df.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(trades_df["cash_remaining"].values, color="#42A5F5", linewidth=1.2)
            ax.set_title("Paper Trading — Cash Balance Over Trades")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Cash (USD)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        if "price" in trades_df.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(trades_df["price"].values, color="#78909C", linewidth=1.0, label="BTC Price")
            if not buy_df.empty:
                ax.scatter(buy_df.index, buy_df["price"], color="#66BB6A", marker="^", s=50, label="Buy", zorder=5)
            if not sell_df.empty:
                ax.scatter(sell_df.index, sell_df["price"], color="#EF5350", marker="v", s=50, label="Sell", zorder=5)
            ax.set_title("Paper Trades on Price")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Trade log"):
            st.dataframe(trades_df, use_container_width=True)

    else:
        st.info(
            f"No paper trade log found at '{paper_log}'. Run `python paper_trade.py` to generate data."
        )

    # Backtest result
    st.subheader("Latest Backtest Result")
    bt = _load_backtest_result()
    if bt:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Return", f"{bt.get('total_return', 0):+.2f}%")
        b2.metric("Sharpe Ratio", f"{bt.get('sharpe', 0):.4f}")
        b3.metric("Max Drawdown", f"{bt.get('max_dd', 0) * 100:.2f}%")
        b4.metric("Win Rate", f"{bt.get('win_rate', 0) * 100:.1f}%" if bt.get("win_rate") else "N/A")
    else:
        st.info("No backtest_result.json found. Run `python backtest.py` and save output as JSON.")

# ── Tab 4: Model Comparison ───────────────────────────────────────────────

with tabs[3]:
    st.header("Model Comparison: DQN vs D3QN vs DDPG")

    st.markdown(
        "Compare performance metrics across different agent architectures trained on the same data."
    )

    # Try to load per-model result files
    model_results: dict[str, dict] = {}
    for model_name in ("DQN", "D3QN", "DDPG"):
        path = os.path.join(output_dir, f"{model_name}_result.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    model_results[model_name] = json.load(fh)
            except Exception:
                pass

    if model_results:
        comp_data = []
        for name, res in model_results.items():
            comp_data.append({
                "Model": name,
                "Total Return (%)": f"{res.get('total_return', 0):+.2f}",
                "Sharpe": f"{res.get('sharpe', 0):.4f}",
                "Max Drawdown (%)": f"{res.get('max_dd', 0) * 100:.2f}",
                "Win Rate (%)": f"{res.get('win_rate', 0) * 100:.1f}",
                "Num Trades": res.get("num_trades", "N/A"),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        # Bar chart comparison
        metrics_to_plot = ["Total Return (%)", "Sharpe"]
        comp_df = pd.DataFrame(comp_data).set_index("Model")
        for m in ["Total Return (%)", "Sharpe"]:
            try:
                comp_df[m] = pd.to_numeric(comp_df[m], errors="coerce")
            except Exception:
                pass

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, metric in enumerate(["Total Return (%)", "Sharpe"]):
            if metric in comp_df.columns:
                comp_df[metric].plot(kind="bar", ax=axes[idx], color=["#42A5F5", "#66BB6A", "#FFA726"])
                axes[idx].set_title(metric)
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis="x", rotation=0)
                axes[idx].grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(
            f"No per-model result files found in '{output_dir}'. "
            "Expected files: DQN_result.json, D3QN_result.json, DDPG_result.json.\n\n"
            "Below is an illustrative comparison with synthetic data:"
        )
        # Illustrative table
        illustrative = pd.DataFrame(
            {
                "Model": ["DQN", "D3QN", "DDPG"],
                "Total Return (%)": ["+12.4", "+18.7", "+9.1"],
                "Sharpe": ["1.32", "1.87", "0.94"],
                "Max Drawdown (%)": ["-8.2", "-6.5", "-11.3"],
                "Win Rate (%)": ["54.0", "58.3", "51.2"],
            }
        )
        st.dataframe(illustrative, use_container_width=True)
        st.caption("Illustrative values only — train models and export results to see real comparisons.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "LARSA Evaluation Dashboard | Data refreshes every 30 s | Not financial advice"
)

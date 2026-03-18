"""DeepSeek V3 signal extraction for BTC news articles.

This module calls the DeepSeek chat API to produce structured sentiment and
risk scores for each news article.  Results are written to a CSV file with
periodic checkpointing so that interrupted runs can be resumed.

Typical usage::

    python deepseek_signals.py --input ./data/news_train.csv \\
                               --output ./data/news_with_signals.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any

import pandas as pd
from openai import OpenAI

from config import settings
from exceptions import DataFetchError, SignalError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SENTIMENT_PROMPT_TEMPLATE: str = """
You are a financial news analyst. Your task is to analyze the sentiment of the following news article.

You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:
- "sentiment_score": An integer from 1 (extremely negative) to 5 (extremely positive), where 3 means neutral.
- "confidence_score_sentiment": A float between 0.0 and 1.0, representing your confidence in the sentiment analysis.
- "reasoning_sentiment": A concise, one-sentence explanation for your sentiment score.

---
Here is a perfect example of the output format:
{{
    "sentiment_score": 4,
    "confidence_score_sentiment": 0.95,
    "reasoning_sentiment": "The article reports a significant earnings beat and a positive future outlook, which are strong bullish signals."
}}
---

Now, analyze the following news item and provide ONLY the JSON object as your response.

Title: {title}
Article Text: {text}
"""

RISK_PROMPT_TEMPLATE: str = """
You are a professional cryptocurrency risk analyst. Your task is to analyze the following news article to identify potential risks related to Bitcoin (BTC) or the broader crypto market.

You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:
- "risk_score": An integer from 1 (extremely negative) to 5 (extremely positive), where 3 means neutral.
- "confidence_score_risk": A float between 0.0 and 1.0, representing your confidence in the risk analysis.
- "reasoning_risk": A concise, one-sentence explanation for your risk assessment.

---
Here is a perfect example of the output format for a BTC-related article:
{{
    "risk_score": 4,
    "confidence_score_risk": 0.85,
    "reasoning_risk": "The announcement of new government regulations on crypto transactions introduces significant legal and compliance uncertainty, potentially stifling adoption."
}}
---

Now, analyze the following news item and provide ONLY the JSON object as your response.

Title: {title}
Article Text: {text}
"""


def _build_client() -> OpenAI:
    """Construct the OpenAI-compatible client pointed at the DeepSeek endpoint.

    Returns:
        An :class:`openai.OpenAI` client instance.

    Raises:
        SignalError: If ``DEEPSEEK_API_KEY`` is not configured.
    """
    api_key = settings.deepseek_api_key
    if not api_key:
        raise SignalError(
            "DEEPSEEK_API_KEY is not set. Export it as an environment variable "
            "or add it to a .env file before running signal extraction."
        )
    return OpenAI(api_key=api_key, base_url=settings.deepseek_base_url)


# Lazily initialised module-level client so tests can patch it.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the module-level DeepSeek client, creating it on first call.

    Returns:
        Shared :class:`openai.OpenAI` client instance.

    Raises:
        SignalError: Propagated from :func:`_build_client`.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------------------------------------------------------------------------
# Core API calls
# ---------------------------------------------------------------------------


def _call_api(prompt: str) -> dict[str, Any]:
    """Send a single prompt to the DeepSeek API and parse the JSON response.

    Args:
        prompt: Fully formatted prompt string.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        SignalError: If the API call fails or the response is not valid JSON.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.deepseek_temperature,
            max_tokens=settings.deepseek_max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise SignalError("DeepSeek API request failed.", cause=exc) from exc

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise SignalError(
            f"DeepSeek returned non-JSON content: {content[:200]}", cause=exc
        ) from exc


def _call_with_retry(prompt: str) -> dict[str, Any]:
    """Call the DeepSeek API with exponential-backoff retry.

    Args:
        prompt: Fully formatted prompt string.

    Returns:
        Parsed JSON response dictionary.

    Raises:
        SignalError: After all retry attempts are exhausted.
    """
    backoff = list(settings.retry_backoff_seconds)
    max_retries = settings.max_retries

    for attempt in range(max_retries):
        try:
            return _call_api(prompt)
        except SignalError as exc:
            log.warning(
                "api.retry",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
            )
            if attempt < max_retries - 1:
                wait = backoff[min(attempt, len(backoff) - 1)] + random.uniform(0, 1)
                log.info("api.backoff", wait_seconds=round(wait, 2))
                time.sleep(wait)

    raise SignalError(f"DeepSeek API failed after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def analyze_article_sentiment(title: str, text: str) -> dict[str, Any] | None:
    """Extract a structured sentiment analysis for one news article.

    Calls the DeepSeek API and validates that all required keys are present.
    Returns ``None`` rather than raising if the API result is structurally
    invalid, so the caller can decide how to handle missing values.

    Args:
        title: Title of the news article.
        text: Body text of the news article.

    Returns:
        A dictionary with keys ``sentiment_score`` (int 1-5),
        ``confidence_score_sentiment`` (float 0-1), and
        ``reasoning_sentiment`` (str), or ``None`` if the response is invalid.

    Raises:
        SignalError: If all retry attempts are exhausted.
    """
    prompt = SENTIMENT_PROMPT_TEMPLATE.format(title=title, text=text)
    try:
        data = _call_with_retry(prompt)
    except SignalError:
        log.error("sentiment.failed", title=title[:60])
        return None

    required = {"sentiment_score", "confidence_score_sentiment", "reasoning_sentiment"}
    if not required.issubset(data.keys()):
        log.error(
            "sentiment.missing_keys",
            present=list(data.keys()),
            required=list(required),
            title=title[:60],
        )
        return None

    confidence: float = float(data.get("confidence_score_sentiment", 0))
    if confidence < settings.min_confidence_threshold:
        log.warning(
            "sentiment.low_confidence",
            confidence=round(confidence, 3),
            threshold=settings.min_confidence_threshold,
            title=title[:60],
        )
    return data


def analyze_article_risk(title: str, text: str) -> dict[str, Any] | None:
    """Extract a structured risk analysis for one news article.

    Args:
        title: Title of the news article.
        text: Body text of the news article.

    Returns:
        A dictionary with keys ``risk_score`` (int 1-5),
        ``confidence_score_risk`` (float 0-1), and ``reasoning_risk`` (str),
        or ``None`` if the response is invalid.

    Raises:
        SignalError: If all retry attempts are exhausted.
    """
    prompt = RISK_PROMPT_TEMPLATE.format(title=title, text=text)
    try:
        data = _call_with_retry(prompt)
    except SignalError:
        log.error("risk.failed", title=title[:60])
        return None

    required = {"risk_score", "confidence_score_risk", "reasoning_risk"}
    if not required.issubset(data.keys()):
        log.error(
            "risk.missing_keys",
            present=list(data.keys()),
            required=list(required),
            title=title[:60],
        )
        return None

    confidence: float = float(data.get("confidence_score_risk", 0))
    if confidence < settings.min_confidence_threshold:
        log.warning(
            "risk.low_confidence",
            confidence=round(confidence, 3),
            threshold=settings.min_confidence_threshold,
            title=title[:60],
        )
    return data


def get_sentiment_and_risk(row: pd.Series) -> pd.Series:
    """Run both sentiment and risk analysis for a single article row.

    Args:
        row: A :class:`pandas.Series` with at minimum ``title`` and
            ``article_text`` fields.

    Returns:
        A :class:`pandas.Series` with six analysis fields: ``sentiment_score``,
        ``confidence_score_sentiment``, ``reasoning_sentiment``,
        ``risk_score``, ``confidence_score_risk``, and ``reasoning_risk``.
        Any field whose analysis failed will be ``None``.
    """
    sentiment = analyze_article_sentiment(row["title"], row["article_text"]) or {}
    risk = analyze_article_risk(row["title"], row["article_text"]) or {}

    return pd.Series(
        {
            "sentiment_score": sentiment.get("sentiment_score"),
            "confidence_score_sentiment": sentiment.get("confidence_score_sentiment"),
            "reasoning_sentiment": sentiment.get("reasoning_sentiment"),
            "risk_score": risk.get("risk_score"),
            "confidence_score_risk": risk.get("confidence_score_risk"),
            "reasoning_risk": risk.get("reasoning_risk"),
        }
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be processed due to missing or empty fields.

    Args:
        df: Raw input :class:`pandas.DataFrame`.

    Returns:
        Cleaned :class:`pandas.DataFrame` with a reset integer index.

    Raises:
        DataFetchError: If the resulting DataFrame is empty.
    """
    original_len = len(df)
    log.info("data.clean.start", rows=original_len)

    df = df.dropna(subset=["title", "article_text"])
    df = df.drop_duplicates(subset=["title", "article_text"])
    df = df[
        (df["title"].str.strip() != "") & (df["article_text"].str.strip() != "")
    ]
    df = df.reset_index(drop=True)

    log.info("data.clean.done", original=original_len, remaining=len(df))
    if len(df) == 0:
        raise DataFetchError("No valid rows remain after cleaning the input data.")
    return df


def load_checkpoint(checkpoint_file: str) -> pd.DataFrame | None:
    """Load an existing progress checkpoint if one exists.

    Args:
        checkpoint_file: Filesystem path to the checkpoint CSV.

    Returns:
        A :class:`pandas.DataFrame` of previously processed rows, or ``None``
        if the file does not exist.

    Raises:
        DataFetchError: If the checkpoint file exists but cannot be parsed.
    """
    if not os.path.exists(checkpoint_file):
        return None
    log.info("checkpoint.load", path=checkpoint_file)
    try:
        return pd.read_csv(checkpoint_file)
    except Exception as exc:
        raise DataFetchError(
            f"Could not read checkpoint file '{checkpoint_file}'.", cause=exc
        ) from exc


def save_checkpoint(df: pd.DataFrame, checkpoint_file: str) -> None:
    """Persist a progress snapshot to disk.

    Args:
        df: The current result :class:`pandas.DataFrame` to save.
        checkpoint_file: Destination path for the CSV snapshot.

    Raises:
        DataFetchError: If the file cannot be written.
    """
    try:
        df.to_csv(checkpoint_file, index=False, encoding="utf-8")
        log.info("checkpoint.saved", path=checkpoint_file, rows=len(df))
    except OSError as exc:
        raise DataFetchError(
            f"Failed to write checkpoint '{checkpoint_file}'.", cause=exc
        ) from exc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_news_analysis(input_file: str, output_file: str) -> None:
    """Run the full sentiment-and-risk annotation pipeline.

    Loads *input_file*, resumes from an existing checkpoint if present,
    annotates each row, saves *output_file*, and cleans up the checkpoint.

    Args:
        input_file: Path to the raw news CSV (must have ``title`` and
            ``article_text`` columns).
        output_file: Destination path for the annotated CSV.

    Raises:
        DataFetchError: If the input file cannot be read or is empty after
            cleaning.
        SignalError: Propagated from signal extraction if all retries fail
            for a given row (logged and skipped -- does not abort the run).
    """
    log.info("pipeline.start", input=input_file, output=output_file)

    try:
        news_df = pd.read_csv(input_file)
    except Exception as exc:
        raise DataFetchError(
            f"Cannot read input file '{input_file}'.", cause=exc
        ) from exc

    news_df = clean_and_validate_data(news_df)

    checkpoint_file = f"{output_file.rsplit('.', 1)[0]}_checkpoint.csv"
    processed_df = load_checkpoint(checkpoint_file)

    if processed_df is not None:
        processed_titles: set[str] = set(processed_df["title"].dropna())
        remaining_df = news_df[~news_df["title"].isin(processed_titles)]
        log.info(
            "pipeline.resume",
            already_done=len(processed_df),
            remaining=len(remaining_df),
        )
        if len(remaining_df) == 0:
            log.info("pipeline.already_complete")
            processed_df.to_csv(output_file, index=False, encoding="utf-8")
            return
        result_df = processed_df.copy()
        start_idx = len(processed_df)
    else:
        remaining_df = news_df.copy()
        result_df = news_df.copy()
        start_idx = 0
        log.info("pipeline.fresh", rows=len(remaining_df))

    total_rows = len(news_df)
    for idx, (_, row) in enumerate(remaining_df.iterrows(), start=start_idx):
        log.debug("pipeline.row", row=idx + 1, total=total_rows, title=str(row["title"])[:60])
        try:
            analysis = get_sentiment_and_risk(row)
            for col in analysis.index:
                result_df.loc[result_df["title"] == row["title"], col] = analysis[col]
        except Exception as exc:
            log.error("pipeline.row_failed", row=idx + 1, error=str(exc))
            continue

        if (idx + 1) % settings.checkpoint_interval == 0:
            save_checkpoint(result_df, checkpoint_file)
            log.info("pipeline.progress", completed=idx + 1, total=total_rows)

    try:
        result_df.to_csv(output_file, index=False, encoding="utf-8")
        log.info("pipeline.done", output=output_file, rows=len(result_df))
    except OSError as exc:
        raise DataFetchError(f"Cannot write output file '{output_file}'.", cause=exc) from exc

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        log.info("checkpoint.removed", path=checkpoint_file)

    sentiment_valid = int(result_df["sentiment_score"].notna().sum())
    risk_valid = int(result_df["risk_score"].notna().sum())
    n = len(result_df)
    log.info(
        "pipeline.summary",
        total=n,
        sentiment_valid=sentiment_valid,
        sentiment_pct=round(sentiment_valid / n * 100, 1) if n else 0,
        risk_valid=risk_valid,
        risk_pct=round(risk_valid / n * 100, 1) if n else 0,
    )


def main() -> None:
    """CLI entry point for the news analysis pipeline.

    Raises:
        SystemExit: With a non-zero code if required arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Annotate news articles with DeepSeek sentiment and risk scores."
    )
    parser.add_argument(
        "--input",
        default="./data/news_train.csv",
        help="Input CSV file path (default: ./data/news_train.csv)",
    )
    parser.add_argument(
        "--output",
        default="BTC_1sec_with_sentiment_risk_train.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=settings.min_confidence_threshold,
        help="Minimum confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=settings.checkpoint_interval,
        help="Save checkpoint every N rows (default: %(default)s).",
    )
    args = parser.parse_args()

    # Override singleton settings from CLI flags.
    settings.min_confidence_threshold = args.min_confidence
    settings.checkpoint_interval = args.checkpoint_interval

    if not os.path.exists(args.input):
        log.error("cli.input_missing", path=args.input)
        raise SystemExit(f"Input file not found: {args.input}")

    log.info(
        "cli.config",
        input=args.input,
        output=args.output,
        min_confidence=settings.min_confidence_threshold,
        checkpoint_interval=settings.checkpoint_interval,
    )

    try:
        process_news_analysis(args.input, args.output)
    except Exception as exc:
        log.error("cli.failed", error=str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

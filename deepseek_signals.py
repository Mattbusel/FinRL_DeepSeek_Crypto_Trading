like this import os
import json
import pandas as pd
from openai import OpenAI
import time
import random
import argparse
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_analysis.log'),
        logging.StreamHandler()
    ]
)

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com/v1"
)

# Rate limiting and retry configuration
MAX_RETRIES = 5
RETRY_BACKOFF = [1, 2, 4, 8, 16]  # Exponential backoff in seconds
CHECKPOINT_INTERVAL = 10  # Save progress every N rows
MIN_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score to accept

##### Prompt Engineering
# A robust prompt that requests a JSON object with all the required fields.
# It includes a high-quality example to make the output very reliable.
SENTIMENT_PROMPT_TEMPLATE = """
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

RISK_PROMPT_TEMPLATE = """
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

def call_with_retry(fn, *args, **kwargs) -> Optional[Dict[Any, Any]]:
    """
    Wrapper function to add retry logic with exponential backoff.
    
    Args:
        fn: Function to call
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function call or None if all retries fail
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            if result:
                return result
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_BACKOFF[attempt] + random.uniform(0, 1)
            logging.info(f"Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
    
    logging.error("Max retries exceeded.")
    return None

def analyze_article_sentiment(title: str, text: str) -> Optional[Dict[str, Any]]:
    """
    Calls the DeepSeek API to get a structured sentiment analysis.

    Args:
        title: The title of the news article.
        text: The content of the news article.

    Returns:
        A dictionary containing 'sentiment_score', 'confidence_score_sentiment', and 'reasoning_sentiment',
        or None if an error occurs.
    """
    formatted_prompt = SENTIMENT_PROMPT_TEMPLATE.format(title=title, text=text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,  # Deterministic output
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        analysis_data = json.loads(response_content)

        required_keys = ["sentiment_score", "confidence_score_sentiment", "reasoning_sentiment"]
        if all(key in analysis_data for key in required_keys):
            # Check confidence threshold
            confidence = analysis_data.get("confidence_score_sentiment", 0)
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                logging.warning(f"Low confidence ({confidence:.2f}) for sentiment analysis of: '{title[:50]}...'")
            return analysis_data
        else:
            logging.error(f"Missing required keys in sentiment response for title: '{title[:50]}...'")
            return None

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON for sentiment analysis of title: '{title[:50]}...'. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"API error occurred for sentiment analysis of title '{title[:50]}...': {e}")
        return None

def analyze_article_risk(title: str, text: str) -> Optional[Dict[str, Any]]:
    """
    Calls the DeepSeek API to get a structured risk analysis.

    Args:
        title: The title of the news article.
        text: The content of the news article.

    Returns:
        A dictionary containing 'risk_score', 'confidence_score_risk', and 'reasoning_risk',
        or None if an error occurs.
    """
    formatted_prompt = RISK_PROMPT_TEMPLATE.format(title=title, text=text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,  # Deterministic output
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        analysis_data = json.loads(response_content)

        required_keys = ["risk_score", "confidence_score_risk", "reasoning_risk"]
        if all(key in analysis_data for key in required_keys):
            # Check confidence threshold
            confidence = analysis_data.get("confidence_score_risk", 0)
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                logging.warning(f"Low confidence ({confidence:.2f}) for risk analysis of: '{title[:50]}...'")
            return analysis_data
        else:
            logging.error(f"Missing required keys in risk response for title: '{title[:50]}...'")
            return None

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON for risk analysis of title: '{title[:50]}...'. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"API error occurred for risk analysis of title '{title[:50]}...': {e}")
        return None

def get_sentiment_and_risk(row: pd.Series) -> pd.Series:
    """
    Combined function to get both sentiment and risk analysis in one pass.
    
    Args:
        row: A pandas Series containing 'title' and 'article_text'
    
    Returns:
        A pandas Series with sentiment and risk analysis results
    """
    # Get sentiment analysis with retry logic
    sentiment = call_with_retry(analyze_article_sentiment, row['title'], row['article_text']) or {}
    
    # Get risk analysis with retry logic
    risk = call_with_retry(analyze_article_risk, row['title'], row['article_text']) or {}
    
    return pd.Series({
        "sentiment_score": sentiment.get("sentiment_score"),
        "confidence_score_sentiment": sentiment.get("confidence_score_sentiment"),
        "reasoning_sentiment": sentiment.get("reasoning_sentiment"),
        "risk_score": risk.get("risk_score"),
        "confidence_score_risk": risk.get("confidence_score_risk"),
        "reasoning_risk": risk.get("reasoning_risk"),
    })

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input data.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    logging.info(f"Original dataset size: {len(df)} rows")
    
    # Remove rows with missing title or article_text
    df = df.dropna(subset=['title', 'article_text'])
    logging.info(f"After removing NaN values: {len(df)} rows")
    
    # Remove duplicates based on title and article_text
    df = df.drop_duplicates(subset=['title', 'article_text'])
    logging.info(f"After removing duplicates: {len(df)} rows")
    
    # Remove rows with empty strings
    df = df[(df['title'].str.strip() != '') & (df['article_text'].str.strip() != '')]
    logging.info(f"After removing empty strings: {len(df)} rows")
    
    return df.reset_index(drop=True)

def load_checkpoint(checkpoint_file: str) -> Optional[pd.DataFrame]:
    """
    Load existing checkpoint file if it exists.
    
    Args:
        checkpoint_file: Path to checkpoint file
    
    Returns:
        DataFrame with processed data or None if file doesn't exist
    """
    if os.path.exists(checkpoint_file):
        logging.info(f"Found existing checkpoint: {checkpoint_file}")
        return pd.read_csv(checkpoint_file)
    return None

def save_checkpoint(df: pd.DataFrame, checkpoint_file: str):
    """
    Save current progress to checkpoint file.
    
    Args:
        df: DataFrame to save
        checkpoint_file: Path to save file
    """
    df.to_csv(checkpoint_file, index=False, encoding='utf-8')
    logging.info(f"Checkpoint saved: {checkpoint_file}")

def process_news_analysis(input_file: str, output_file: str):
    """
    Main processing function with checkpointing and error handling.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Load and clean data
    logging.info(f"Loading data from: {input_file}")
    news_df = pd.read_csv(input_file)
    news_df = clean_and_validate_data(news_df)
    
    if len(news_df) == 0:
        logging.error("No valid data to process after cleaning.")
        return
    
    # Check for existing checkpoint
    checkpoint_file = f"{output_file.rsplit('.', 1)[0]}_checkpoint.csv"
    processed_df = load_checkpoint(checkpoint_file)
    
    if processed_df is not None:
        # Resume from checkpoint
        processed_titles = set(processed_df["title"].dropna())
        remaining_df = news_df[~news_df["title"].isin(processed_titles)]
        logging.info(f"Resuming from checkpoint. {len(remaining_df)} rows remaining to process.")
        
        if len(remaining_df) == 0:
            logging.info("All data already processed. Saving final output.")
            processed_df.to_csv(output_file, index=False, encoding='utf-8')
            return
        
        # Initialize result DataFrame with existing processed data
        result_df = processed_df.copy()
        start_idx = len(processed_df)
    else:
        # Start fresh
        remaining_df = news_df.copy()
        result_df = news_df.copy()
        start_idx = 0
        logging.info(f"Starting fresh analysis of {len(remaining_df)} rows")
    
    # Process remaining data
    total_rows = len(news_df)
    
    for idx, (_, row) in enumerate(remaining_df.iterrows(), start=start_idx):
        try:
            logging.info(f"Processing row {idx + 1}/{total_rows}: {row['title'][:50]}...")
            
            # Get combined analysis
            analysis_result = get_sentiment_and_risk(row)
            
            # Add analysis results to the result DataFrame
            if idx < len(result_df):
                # Update existing row
                for col in analysis_result.index:
                    result_df.loc[result_df['title'] == row['title'], col] = analysis_result[col]
            else:
                # Append new row (shouldn't happen in normal flow, but safety check)
                new_row = row.copy()
                for col in analysis_result.index:
                    new_row[col] = analysis_result[col]
                result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)
            
            # Save checkpoint periodically
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(result_df, checkpoint_file)
                logging.info(f"Progress: {idx + 1}/{total_rows} rows completed")
            
        except Exception as e:
            logging.error(f"Error processing row {idx + 1}: {e}")
            continue
    
    # Save final results
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"Successfully saved final results to: {output_file}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logging.info("Checkpoint file cleaned up")
    
    # Log summary statistics
    logging.info("Analysis Summary:")
    logging.info(f"Total rows processed: {len(result_df)}")
    sentiment_valid = result_df['sentiment_score'].notna().sum()
    risk_valid = result_df['risk_score'].notna().sum()
    logging.info(f"Valid sentiment analyses: {sentiment_valid}/{len(result_df)} ({sentiment_valid/len(result_df)*100:.1f}%)")
    logging.info(f"Valid risk analyses: {risk_valid}/{len(result_df)} ({risk_valid/len(result_df)*100:.1f}%)")

def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description="Analyze news articles for sentiment and risk")
    parser.add_argument("--input", default="./data/news_train.csv", 
                        help="Input CSV file path (default: ./data/news_train.csv)")
    parser.add_argument("--output", default="BTC_1sec_with_sentiment_risk_train.csv", 
                        help="Output CSV file path (default: BTC_1sec_with_sentiment_risk_train.csv)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence threshold (default: 0.3)")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N rows (default: 10)")
    
    args = parser.parse_args()

    # Update global configuration
    global MIN_CONFIDENCE_THRESHOLD, CHECKPOINT_INTERVAL
    MIN_CONFIDENCE_THRESHOLD = args.min_confidence
    CHECKPOINT_INTERVAL = args.checkpoint_interval

    # Validate input file exists
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return

    logging.info("Starting news analysis script")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Minimum confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")
    logging.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")

    try:
        process_news_analysis(args.input, args.output)
        logging.info("Script completed successfully!")
    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        raise

   

if __name__ == "__main__":
    main()

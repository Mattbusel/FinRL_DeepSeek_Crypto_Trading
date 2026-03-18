# Architecture

## System Overview

```
+---------------------+       +--------------------+       +------------------+
|  Reddit / News Feed |  -->  |  DeepSeek V3 API   |  -->  |  Signal Store    |
|  (CSV / live feed)  |       |  deepseek_signals  |       |  (CSV + .npy)    |
+---------------------+       +--------------------+       +------------------+
                                                                    |
                                                                    v
                               +----------------------------+
                               |  RNN Factor Model          |
                               |  seq_run.py / seq_net.py   |
                               |  LSTM + GRU regression     |
                               +----------------------------+
                                            |
                                            v
+-------------------+       +-------------------------------+       +------------------+
|  TradeSimulator   |  <--> |  Ensemble Agent               |  -->  |  Trade Execution |
|  Vectorised env   |       |  D3QN + DoubleDQN + TwinD3QN  |       |  (positions.npy) |
|  1-second BTC     |       |  Majority-vote ensemble       |       |  + metrics       |
+-------------------+       +-------------------------------+       +------------------+
```

## Module Descriptions

### deepseek_signals.py

Calls the DeepSeek V3 chat API for each news article to produce a JSON object
with sentiment score (1-5), risk score (1-5), confidence values, and
one-sentence reasoning. Implements exponential-backoff retry, confidence
thresholding, and CSV checkpointing so interrupted runs resume cleanly.

### seq_data.py / seq_run.py / seq_net.py

`seq_data.py` implements the Alpha101 factor library and converts raw BTC
1-second price/book data into input and label numpy arrays.

`seq_run.py` defines `SeqData` (train/validation split) and the `train_model`
loop using `AdamW` with gradient clipping and early stopping.

`seq_net.py` defines `RnnRegNet`: a parallel LSTM + GRU trunk feeding a shared
MLP head. Output is a per-timestep label prediction (regression).

### task1_ensemble.py

Trains multiple agent classes sequentially under a shared `Config`, then saves
each to a named sub-directory. Exposes `Ensemble._majority_vote` for
tie-breaking.

### task1_eval.py

Loads saved agent checkpoints, runs a single deterministic evaluation episode,
performs cash/BTC accounting step by step, and reports Sharpe ratio, max
drawdown, and return-over-max-drawdown.

### config.py

`LARSASettings` is a `pydantic-settings` `BaseSettings` subclass. Every
hyperparameter (learning rate, batch size, DeepSeek model, slippage, etc.) is
declared here with validation rules and documentation strings. Override any
value by exporting the matching uppercase environment variable.

### exceptions.py

`LARSAError` base class with four specialisations:
- `DataFetchError` - data loading / file I/O failures
- `ModelError` - checkpoint loading / shape mismatches
- `SignalError` - API failures / malformed JSON
- `ConfigError` - invalid or missing configuration

### logger.py

`get_logger(name)` returns a `logging.Logger` configured to emit single-line
JSON records to stdout. Every record includes timestamp, level, logger name,
message, and any extra key-value pairs passed to the logging call.

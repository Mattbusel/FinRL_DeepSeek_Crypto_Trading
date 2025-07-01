<<<<<<< HEAD
# FinRL_DeepSeek_Crypto_Trading
=======

# FinAI Contest Task 1 - FinRL-DeepSeek for Crypto Trading
This task is to develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data. Participants can build upon the FinRL-DeepSeek project (e.g., with new prompts, new ways to inject LLM-processed news signals into the RL agent, new RL algorithms like GRPO) or explore more computationally intensive directions, such as adapting variants of the DeepSeek R1 training method to this crypto trading task.

Participants are encouraged to explore a wide range of strategies for extracting actionable signals from financial news and limit order book (LOB) data. They may employ prompt engineering or advanced LLM-based pipelines to generate signals. External datasets are permitted and welcomed, offering the opportunity to integrate diverse signals such as macroeconomic indicators, crypto wallets activity, or alternative datasets. Partcipants can also experiment with the RNN-based feature engineering, different RL algorithms, and employ ensemble methods. Moreover, teams are encouraged to optimize GPU-accelerated environments.

Participants also need to specify the state space, action space and reward function in the environment. The final model should be able to interact with the provided trading environment.

There is an example of using feature engineering and ensemble method, [FinRL Contest 2025 Task 2 Crypto Trading Ensemble](). 

## Starter Kit Description

This starter kit demonstrates how to use the provided code. We provide you with the LLM-engineered sentiment and risk scores and RNN-generated strong factors as a starting point, but you are strongly encouraged to develop your own factor mining approaches. You are welcome to experiment with various ensemble configurations that yield optimal results.

### LLM-engineered Signals from BTC news

-  `deepseek_signals.py`: Extracts sentiment score and risk score from BTC news using the DeepSeek API. It contains two prompt templates designed to query the API for both sentiment and risk signals. For each news article, the script sends the relevant prompt to DeepSeek V3, receives the extracted signal, and records the confidence and reasoning. The resulting sentiment and risk scores can be used as features or signals for downstream trading models or analysis.

### Supervised Training of Deep Learning Recurrent Networks

- `seq_data.py`: Reads BTC's CSV data and generates Alpha101 weak factors.
  - Function `convert_csv_to_level5_csv`: Reads a CSV file into a DataFrame, then extracts the required level5 data, and saves it back as a CSV.
  - Function `convert_btc_csv_to_btc_npy`: Reads a CSV file, saves it as an array, and linearly transforms it to between ±1 using min-max values, saving as an npy. (z-score)

- `seq_net.py`: Feeds a time series into a recurrent network `(LSTM+GRU + MLP)` to predict another time series as the label.
  - Class `RnnRegNet`: Processes `input_seq → Concatenate(LSTM, GRU) → RegressionMLP → label_seq`

- `seq_run.py`: Inputs Alpha101 weak factor series to train the deep learning recurrent network, predicting future price movement labels.
  - Class `SeqData`: Prepares the input and output sequences for training the neural network, using the function `sample_for_train` to randomly cut sequences for training.
  - Function `train_model`: Uses the condition "number of steps without improvement in loss reaches the set limit" as the criterion for early stopping during training.

- `seq_record.py`: Records the training process logs and plots the loss function graph.
  - Class `Evaluator`: Evaluates model performance during training, tracking accuracy and loss values.

### Reinforcement Learning DQN Algorithm Training in a Market Replay Simulator

- `trade_simulator.py`: Contains a market replay simulator for single commodity.
  - Class `TradeSimulator`: A training-focused market replay simulator, complying with the older gym-style API requirements.
  - Class `EvalTradeSimulator`: A market replay simulator for evaluation.

- `erl_config.py`: Configuration file for reinforcement learning training.

- `erl_replay_buffer.py`: Serves as a training dataset for the reinforcement learning market replay simulator.

- `erl_agent.py`: Contains the DQN class algorithm for reinforcement learning.

- `erl_net.py`: Neural network structures used in the reinforcement learning algorithm.

- `erl_run.py`: Loads the simulator and trains the FinRL agent.

- `erl_evaluator.py`: Evaluates the performance of the reinforcement learning agent.

- `metrics.py`: Contains some metrics for evaluation.

- `task1_ensemble.py`: This file contains code that trains multiple models and then saves them to be tested during evaluation.

- `task1_eval.py`: This file contains code that loads your ensemble and simulates trading over a validation dataset. You may create this validation dataset by holding out a part of the training data.

## Dataset

A dataset containing second-level Limit Order Book (LOB) data and financial news for Bitcoin is provided. Please download [here](https://drive.google.com/drive/folders/1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp?usp=sharing). All of the datasets required to train DRL agents are in the data directory, please download this into the Task 1 starter dir `data/` folder

The dataset contains the following file: 
- `BTC_news_train.csv`: BTC news with corresponding DeepSeek V3 engineered sentiment scores and risk scores.
- `BTC_1sec_with_sentiment_risk_train.csv`: The merged and time-aligned version of the previous two datasets.

The `BTC_1sec_with_sentiment_risk_train.csv` contains all data used to train the RNN model and FinRL agent. 

## Setup

1. Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Download BTC datasets into data directory

4. Generate Technical Factors Alpha101 Describing the Market

Run `seq_data.py`'s convert_btc_csv_to_btc_npy:
- col1: AlphaID from Alpha1 to Alpha101
- col2: Used time (second) total
- col3: Used time (second) of single Alpha
- col4: alpha shape
- col5: number of nan `nan_rate= nan_number/alpha.shape[0]`

5. Supervised Learning Training Loop Network to Aggregate Multiple Factor Sequences into Fewer Strong Signal Factors

Run `seq_run.py`. Remember to set your GPU_ID, e.g.:
```
python3 seq_run.py 0
```

6. Train FinRL Agents

Run `erl_run.py` for a single agent. Or run `task1_ensemble.py` for ensemble methods. Remember to set your GPU_ID, e.g.:
```
python3 erl_run.py 0
python3 task1_ensemble.py 0
```

## Evaluation

The initial cash is $1 million.

For evaluation, we will run your agents on a test set and compare the results using metrics cumulative return, sharpe ratio, and maximum drawdown. 

We provide an evaluation template that you may use to test your ensemble models. You may change `"predict_ary_path": "BTC_1sec_predict.npy"` in the env_args object to point to a validation subset of the predict ary path which will let you test your model on out of sample data. 

The evaluation kit is in `task1_eval.py` and loads your models then uses a simple voting scheme to perform ensemble actions. As mentioned above, if you change the voting scheme, please be sure to submit your environment code. You may change following:
- `_ensemble_action` you may substitute this for another ensemble action function. Please submit your action functions so that we may use them to evaluate your models.
- `save_path`: this is the path of your saved models for the ensemble saved in the task1_train file. 
- `dataset_path`: please provide a path to a validation dataset. We recommend using a subset of `BTC_1sec_predict.npy` for validation.
- `args.net_dims`: we use a default setting of (128,128,128) but you may choose to use different models. Please provide your settings so that we can evaluate them properly.


## Submission

Please submit all your models and the scripts to load and test the models.

Please provide a readme that describes your submission and explains important things to note when running it so that we can ensure we run your submission as it was intended.

1. You are free to apply any method for extracting signals from financial news, factor mining and ensemble learning. We strongly encourage innovation in all areas, especially in designing your own LLM-engineered signals. (For example, You can add new agents, use different ensemble algorithms, create novel LLM-engineered signals, adjust hyperparameters, etc.) The code provided is just to help get started.

2. You are not required to stick to the signals extraction and factors selection we provide. But for evaluation purpose, please make sure that your new signals from news or technical factors, if any, can be calculated based on the unseen data. Please include this code and state clearly in readme.

3. We will use the provided environment to evaluate. So it is not encouraged to change the basic settings in the environment for fairness, such as transaction costs. However, you can change the envionrment by adding new mechanisms, such as new stop-loss mechanisms. You can fully utilize the environment settings and the massively parallel simulation.

4. To encourage innovation, if you want to add new mechanisms or use 
the unused settings (e.g. short sell, different voting mechanisms for the ensemble) in the environment, please also submit your environment, ensure it works with your agent for evaluation, and describe the new changes in the readme.

```
├── finrl-contest-task-1
│ ├── trained_models # Your trained component agent weights.
| ├── factor_mining # Your code files for the factor mining stage
│ ├── task1_ensemble.py # File for implementing the ensemble method 
│ ├── task1_eval.py # a template evaluation file. Please submit your evaluation code as well.
│ ├── trade_simulator.py # File for the environment. Please submit it if you modified the provided env.
│ ├── README.md # File to explain the your code
│ ├── requirements.txt # Have it if adding any new packages
│ ├── And any additional scripts you create
```
>>>>>>> 6dcf513f (Initial commit - FinRL DeepSeek Crypto Trading baseline setup)

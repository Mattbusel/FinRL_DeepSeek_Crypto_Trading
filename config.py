"""Centralised configuration for the LARSA trading system.

All hyperparameters and external-service settings are declared here.
Values are read from environment variables when present; the defaults shown
below are used otherwise.  Import :data:`settings` in every module instead of
hard-coding magic numbers.

Example::

    from config import settings
    print(settings.deepseek_model)
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LARSASettings(BaseSettings):
    """Validated application settings sourced from environment variables.

    All field names correspond to upper-cased environment variable names
    (e.g. ``DEEPSEEK_API_KEY``, ``BATCH_SIZE``).

    Raises:
        ConfigError: If a required variable is absent or a value is out of
            range (validation happens at import time via
            :func:`pydantic.field_validator`).
    """

    # --- DeepSeek API ---
    deepseek_api_key: str = Field(
        default="",
        description="API key for the DeepSeek inference endpoint.",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="Base URL of the DeepSeek API.",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="Model identifier passed to the DeepSeek chat endpoint.",
    )
    deepseek_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for DeepSeek completions (0 = deterministic).",
    )
    deepseek_max_tokens: int = Field(
        default=300,
        ge=1,
        description="Maximum tokens in a single DeepSeek completion.",
    )

    # --- Signal extraction ---
    max_retries: int = Field(
        default=5,
        ge=1,
        description="Maximum number of API call attempts before giving up.",
    )
    retry_backoff_seconds: list[float] = Field(
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        description="Exponential backoff wait times (seconds) between retries.",
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save a progress checkpoint every N processed rows.",
    )
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Discard signal analyses below this confidence score.",
    )

    # --- RNN sequence model ---
    rnn_batch_size: int = Field(
        default=256,
        ge=1,
        description="Number of samples per RNN training mini-batch.",
    )
    rnn_mid_dim: int = Field(
        default=128,
        ge=1,
        description="Hidden-layer width of the RNN regression network.",
    )
    rnn_num_layers: int = Field(
        default=4,
        ge=1,
        description="Number of stacked recurrent layers.",
    )
    rnn_epochs: int = Field(
        default=256,
        ge=1,
        description="Number of training epochs for the RNN.",
    )
    rnn_warmup_dim: int = Field(
        default=64,
        ge=0,
        description="Sequence steps used for RNN hidden-state warm-up (no loss computed).",
    )
    rnn_valid_gap: int = Field(
        default=128,
        ge=1,
        description="Validate every N training steps.",
    )
    rnn_num_patience: int = Field(
        default=8,
        ge=1,
        description="Early-stopping patience (steps without improvement).",
    )
    rnn_weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="L2 regularisation coefficient for AdamW.",
    )
    rnn_learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Initial learning rate for AdamW.",
    )
    rnn_clip_grad_norm: float = Field(
        default=2.0,
        gt=0.0,
        description="Gradient clipping threshold.",
    )
    rnn_train_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Fraction of sequence data reserved for training.",
    )

    # --- RL agent ---
    rl_learning_rate: float = Field(
        default=2e-6,
        gt=0.0,
        description="Learning rate for the RL actor/critic networks.",
    )
    rl_batch_size: int = Field(
        default=512,
        ge=1,
        description="Mini-batch size for RL network updates.",
    )
    rl_gamma: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description="Discount factor for future rewards.",
    )
    rl_explore_rate: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Epsilon-greedy exploration rate.",
    )
    rl_soft_update_tau: float = Field(
        default=2e-6,
        gt=0.0,
        description="Soft target-network update coefficient.",
    )
    rl_state_value_tau: float = Field(
        default=0.01,
        ge=0.0,
        description="State-value normalisation tau.",
    )
    rl_net_dims: list[int] = Field(
        default=[128, 128, 128],
        description="Hidden-layer widths of the RL policy/value network.",
    )
    rl_break_step: int = Field(
        default=32,
        ge=1,
        description="Stop RL training after this many environment steps.",
    )

    # --- Trade simulation ---
    num_sims: int = Field(
        default=4096,
        ge=1,
        description="Number of parallel simulation environments during training.",
    )
    num_ignore_step: int = Field(
        default=60,
        ge=0,
        description="Initial steps ignored at the start of each episode.",
    )
    max_position: int = Field(
        default=1,
        ge=1,
        description="Maximum absolute BTC position the agent may hold.",
    )
    step_gap: int = Field(
        default=2,
        ge=1,
        description="Number of 1-second ticks aggregated into one environment step.",
    )
    slippage: float = Field(
        default=7e-7,
        ge=0.0,
        description="Per-trade slippage as a fraction of price.",
    )
    starting_cash: float = Field(
        default=1e6,
        gt=0.0,
        description="Initial cash balance for evaluation episodes.",
    )

    # --- Paths ---
    data_dir: str = Field(
        default="./data",
        description="Directory containing price and news CSV/NPY files.",
    )
    output_dir: str = Field(
        default="./output",
        description="Directory for model checkpoints and evaluation artefacts.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{value}'")
        return upper

    model_config = {"env_prefix": "", "case_sensitive": False}


# Module-level singleton consumed by all other modules.
settings = LARSASettings()

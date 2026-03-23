"""Transformer-based price predictor for BTC OHLCV sequences.

Uses a standard encoder-only Transformer to predict the next bar's return
distribution (mean + std) from the last 60 bars of OHLCV + technical
indicators.  The output is a Gaussian parameterisation that can be consumed
directly by the RL agent as an additional signal.

Architecture
------------
* Input projection: linear embedding of each time-step feature vector
* Positional encoding: sinusoidal, shape (seq_len, d_model)
* 4 x TransformerEncoderLayer (8 heads, d_model=128, d_ff=512, dropout=0.1)
* Output head: 2-unit linear (mu, log_sigma) over the CLS-style last token

Example::

    from transformer_predictor import TransformerPricePredictor, PredictorSignal

    predictor = TransformerPricePredictor(feature_dim=10)
    signal = predictor.predict(sequence_tensor)   # sequence_tensor: (60, 10)
    print(signal.mean_return, signal.std_return, signal.confidence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import (degrade gracefully for environments without GPU)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictorSignal:
    """Output of :class:`TransformerPricePredictor`.

    Attributes:
        mean_return: Predicted next-bar log-return (float).
        std_return: Predicted uncertainty (1-sigma, float > 0).
        confidence: Inverse-uncertainty proxy in [0, 1].
        direction: +1 (bullish) / -1 (bearish) / 0 (neutral).
    """

    mean_return: float
    std_return: float
    confidence: float
    direction: int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Add positional encoding to *x* (B, T, d_model)."""
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class _TransformerPricePredictorNet(nn.Module):
        """Core Transformer encoder network.

        Args:
            feature_dim: Number of input features per time-step.
            d_model: Internal embedding dimension (default 128).
            nhead: Number of attention heads (default 8).
            num_layers: Number of encoder layers (default 4).
            d_ff: Feed-forward hidden dimension (default 512).
            dropout: Dropout probability (default 0.1).
        """

        def __init__(
            self,
            feature_dim: int,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
            d_ff: int = 512,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(feature_dim, d_model)
            self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_head = nn.Linear(d_model, 2)  # (mu, log_sigma)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: Input tensor of shape (B, T, feature_dim).

            Returns:
                Tensor of shape (B, 2) — columns are (mu, log_sigma).
            """
            x = self.input_proj(x)          # (B, T, d_model)
            x = self.pos_enc(x)             # (B, T, d_model)
            x = self.encoder(x)             # (B, T, d_model)
            last = x[:, -1, :]             # CLS-style: use last time step
            return self.output_head(last)   # (B, 2)


class TransformerPricePredictor:
    """High-level wrapper around :class:`_TransformerPricePredictorNet`.

    Manages device placement, weight loading, and inference.  When PyTorch is
    unavailable, :meth:`predict` returns a neutral zero-signal stub so the rest
    of the pipeline continues to function.

    Args:
        feature_dim: Number of input features per bar (must match training data).
        seq_len: Expected input sequence length (default 60 bars).
        d_model: Transformer embedding dimension (default 128).
        nhead: Number of attention heads (default 8).
        num_layers: Number of Transformer encoder layers (default 4).
        d_ff: Feed-forward hidden size (default 512).
        dropout: Dropout rate used during training (default 0.1).
        device: ``"cpu"`` or ``"cuda"``; auto-detected when not supplied.
        weights_path: Optional path to a saved ``state_dict`` (``.pt`` file).
    """

    def __init__(
        self,
        feature_dim: int = 10,
        seq_len: int = 60,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self._net = None

        if not _TORCH_AVAILABLE:
            log.warning("transformer_predictor.torch_unavailable", feature_dim=feature_dim)
            return

        import torch  # noqa: F811
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net = _TransformerPricePredictorNet(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        ).to(self._device)

        if weights_path:
            self._load_weights(weights_path)
        self._net.eval()
        log.info(
            "transformer_predictor.ready",
            feature_dim=feature_dim,
            seq_len=seq_len,
            device=str(self._device),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, sequence: "np.ndarray | torch.Tensor") -> PredictorSignal:
        """Predict the next bar's return distribution.

        Args:
            sequence: Array of shape (seq_len, feature_dim) representing the
                last *seq_len* bars of OHLCV + technical indicators.

        Returns:
            :class:`PredictorSignal` with mean_return, std_return, confidence,
            and direction.
        """
        if self._net is None:
            return PredictorSignal(mean_return=0.0, std_return=1.0, confidence=0.0, direction=0)

        import torch  # noqa: F811

        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float32)

        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)  # (1, T, F)

        sequence = sequence.to(self._device)

        with torch.no_grad():
            out = self._net(sequence)          # (1, 2)
            mu = float(out[0, 0].cpu())
            log_sigma = float(out[0, 1].cpu())

        std = math.exp(max(log_sigma, -4.0))   # clamp to avoid near-zero std
        # Confidence: inversely proportional to uncertainty, mapped to [0,1]
        confidence = float(1.0 / (1.0 + std))
        direction = 1 if mu > 0.001 else (-1 if mu < -0.001 else 0)

        signal = PredictorSignal(
            mean_return=mu,
            std_return=std,
            confidence=confidence,
            direction=direction,
        )
        log.debug(
            "transformer_predictor.signal",
            mu=round(mu, 6),
            std=round(std, 6),
            confidence=round(confidence, 4),
            direction=direction,
        )
        return signal

    def train_step(
        self,
        sequences: "torch.Tensor",
        targets: "torch.Tensor",
        optimizer: "torch.optim.Optimizer",
    ) -> float:
        """Single supervised training step (Gaussian NLL loss).

        Args:
            sequences: Batch of input sequences, shape (B, T, feature_dim).
            targets: Next-bar log-returns, shape (B,).
            optimizer: PyTorch optimizer.

        Returns:
            Scalar loss value for this step.
        """
        if self._net is None:
            return 0.0

        import torch  # noqa: F811

        self._net.train()
        sequences = sequences.to(self._device)
        targets = targets.to(self._device)

        out = self._net(sequences)          # (B, 2)
        mu = out[:, 0]
        log_sigma = out[:, 1]

        # Gaussian negative log-likelihood: NLL = log_sigma + 0.5*((y-mu)/sigma)^2
        sigma = torch.exp(log_sigma.clamp(-4, 4))
        nll = log_sigma + 0.5 * ((targets - mu) / sigma) ** 2
        loss = nll.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        optimizer.step()

        self._net.eval()
        return float(loss.item())

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        if self._net is None:
            return
        import torch  # noqa: F811
        torch.save(self._net.state_dict(), path)
        log.info("transformer_predictor.saved", path=path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_weights(self, path: str) -> None:
        import torch  # noqa: F811
        try:
            state = torch.load(path, map_location=self._device)
            self._net.load_state_dict(state)
            log.info("transformer_predictor.weights_loaded", path=path)
        except Exception as exc:
            log.warning("transformer_predictor.weights_load_failed", path=path, error=str(exc))


# ---------------------------------------------------------------------------
# Feature builder helper
# ---------------------------------------------------------------------------

def build_feature_sequence(
    ohlcv: "np.ndarray",
    indicators: Optional["np.ndarray"] = None,
    seq_len: int = 60,
) -> "np.ndarray":
    """Construct a normalised (seq_len, feature_dim) input array.

    Args:
        ohlcv: Array of shape (N, 5) with columns [open, high, low, close, volume].
        indicators: Optional array of shape (N, k) with technical indicator values.
            Concatenated column-wise with *ohlcv*.
        seq_len: Number of bars to return (last *seq_len* rows used).

    Returns:
        Normalised numpy array of shape (seq_len, feature_dim).

    Raises:
        ValueError: If *ohlcv* has fewer than *seq_len* rows.
    """
    if len(ohlcv) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} bars, got {len(ohlcv)}"
        )

    window = ohlcv[-seq_len:]
    if indicators is not None:
        ind_window = indicators[-seq_len:]
        window = np.concatenate([window, ind_window], axis=1)

    # Z-score normalise each column independently
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    return ((window - mean) / std).astype(np.float32)

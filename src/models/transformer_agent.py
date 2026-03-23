"""Transformer-based trading agent (pure Python/numpy-free implementation)."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


class MultiHeadAttention:
    """Scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 42) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self._rng = random.Random(seed)
        self._init_weights()

    def _rand_matrix(self, rows: int, cols: int) -> List[List[float]]:
        scale = math.sqrt(2.0 / (rows + cols))
        return [
            [self._rng.gauss(0.0, scale) for _ in range(cols)]
            for _ in range(rows)
        ]

    def _init_weights(self) -> None:
        d = self.d_model
        self.W_Q = self._rand_matrix(d, d)
        self.W_K = self._rand_matrix(d, d)
        self.W_V = self._rand_matrix(d, d)
        self.W_O = self._rand_matrix(d, d)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        rows, inner, cols = len(A), len(B), len(B[0])
        C = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for k in range(inner):
                if A[i][k] == 0.0:
                    continue
                for j in range(cols):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @staticmethod
    def _matvec(M: List[List[float]], v: List[float]) -> List[float]:
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

    def softmax(self, x: List[float]) -> List[float]:
        max_x = max(x) if x else 0.0
        exps = [math.exp(xi - max_x) for xi in x]
        total = sum(exps)
        return [e / total for e in exps] if total > 0 else exps

    def attention(
        self,
        Q: List[List[float]],
        K: List[List[float]],
        V: List[List[float]],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Scaled dot-product attention. Returns (output, weights)."""
        scale = math.sqrt(self.d_k) if self.d_k > 0 else 1.0
        seq_q = len(Q)
        seq_k = len(K)

        # Scores: (seq_q, seq_k)
        scores: List[List[float]] = []
        for i in range(seq_q):
            row = []
            for j in range(seq_k):
                dot = sum(Q[i][d] * K[j][d] for d in range(len(Q[i])))
                row.append(dot / scale)
            scores.append(row)

        # Softmax over key dimension.
        weights = [self.softmax(row) for row in scores]

        # Output: weighted sum of V.
        d_v = len(V[0]) if V else 0
        output: List[List[float]] = []
        for i in range(seq_q):
            out_row = [0.0] * d_v
            for j in range(seq_k):
                for d in range(d_v):
                    out_row[d] += weights[i][j] * V[j][d]
            output.append(out_row)

        return output, weights

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """Multi-head attention forward pass."""
        # Project Q, K, V.
        Q = [self._matvec(self.W_Q, vec) for vec in x]
        K = [self._matvec(self.W_K, vec) for vec in x]
        V = [self._matvec(self.W_V, vec) for vec in x]

        # Single-head attention (simplified multi-head via full d_model).
        attn_out, _ = self.attention(Q, K, V)

        # Output projection.
        return [self._matvec(self.W_O, vec) for vec in attn_out]


class TransformerLayer:
    """Single transformer encoder layer: attention + residual + FF + residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        rng = random.Random(seed + 1)

        def rand_mat(r: int, c: int) -> List[List[float]]:
            scale = math.sqrt(2.0 / (r + c))
            return [[rng.gauss(0.0, scale) for _ in range(c)] for _ in range(r)]

        self.W1 = rand_mat(ff_dim, d_model)
        self.b1 = [0.0] * ff_dim
        self.W2 = rand_mat(d_model, ff_dim)
        self.b2 = [0.0] * d_model

    def feed_forward(self, x: List[float]) -> List[float]:
        """Two-layer MLP with ReLU."""
        # Layer 1.
        h = [
            max(0.0, sum(self.W1[i][j] * x[j] for j in range(len(x))) + self.b1[i])
            for i in range(self.ff_dim)
        ]
        # Layer 2.
        return [
            sum(self.W2[i][j] * h[j] for j in range(self.ff_dim)) + self.b2[i]
            for i in range(self.d_model)
        ]

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """Attention + residual, then FF + residual."""
        attn_out = self.attention.forward(x)
        # Residual 1.
        after_attn = [
            [x[i][d] + attn_out[i][d] for d in range(self.d_model)]
            for i in range(len(x))
        ]
        # FF + residual 2.
        result = []
        for vec in after_attn:
            ff_out = self.feed_forward(vec)
            result.append([vec[d] + ff_out[d] for d in range(self.d_model)])
        return result


class TransformerTradingAgent:
    """Transformer-based trading agent producing a scalar action in [-1, 1]."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.d_model = d_model
        rng = random.Random(seed)

        # Embedding projection.
        scale = math.sqrt(2.0 / (input_dim + d_model))
        self.W_embed = [
            [rng.gauss(0.0, scale) for _ in range(input_dim)]
            for _ in range(d_model)
        ]
        self.b_embed = [0.0] * d_model

        self.layers = [
            TransformerLayer(d_model, num_heads, d_model * 4, seed=seed + i)
            for i in range(num_layers)
        ]

        # Output head: d_model → 1 scalar.
        scale_out = math.sqrt(2.0 / d_model)
        self.W_out = [rng.gauss(0.0, scale_out) for _ in range(d_model)]
        self.b_out = 0.0

    def embed(self, features: List[float]) -> List[float]:
        """Linear projection: input_dim → d_model."""
        return [
            sum(self.W_embed[i][j] * features[j] for j in range(self.input_dim))
            + self.b_embed[i]
            for i in range(self.d_model)
        ]

    def softmax(self, x: List[float]) -> List[float]:
        max_x = max(x) if x else 0.0
        exps = [math.exp(xi - max_x) for xi in x]
        total = sum(exps)
        return [e / total for e in exps] if total > 0 else exps

    def forward(self, sequence: List[List[float]]) -> float:
        """Embed → transformer layers → mean pool → tanh scalar action."""
        if not sequence:
            return 0.0
        embedded = [self.embed(step) for step in sequence]
        hidden = embedded
        for layer in self.layers:
            hidden = layer.forward(hidden)
        # Mean pool over sequence.
        d = self.d_model
        pooled = [
            sum(hidden[t][i] for t in range(len(hidden))) / len(hidden)
            for i in range(d)
        ]
        # Linear → tanh.
        raw = sum(self.W_out[i] * pooled[i] for i in range(d)) + self.b_out
        return math.tanh(raw)

    def predict_action(self, market_data: List[Dict]) -> float:
        """Extract features (close, volume, returns) and run forward."""
        if not market_data:
            return 0.0
        features_seq = []
        prev_close = None
        for bar in market_data:
            close = float(bar.get("close", 0.0))
            volume = float(bar.get("volume", 0.0))
            ret = (close - prev_close) / prev_close if prev_close and prev_close != 0 else 0.0
            prev_close = close
            features_seq.append([close, volume, ret])
        return self.forward(features_seq)

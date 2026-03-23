"""
Self-attention agent over market history.
Pure Python implementation — no numpy/torch dependency.
"""
import math
import random


def matrix_multiply(A, B):
    """Pure Python matrix multiplication. A: (m x k), B: (k x n) -> (m x n)."""
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            val = 0.0
            for p in range(k):
                val += A[i][p] * B[p][j]
            C[i][j] = val
    return C


def softmax_rows(M):
    """Row-wise softmax on matrix M."""
    result = []
    for row in M:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.
    Q: (seq_len x d_k), K: (seq_len x d_k), V: (seq_len x d_v)
    Returns output: (seq_len x d_v)
    """
    seq_len = len(Q)
    d_k = len(Q[0])
    scale = math.sqrt(d_k) if d_k > 0 else 1.0

    # Scores = Q * K^T / sqrt(d_k): (seq_len x seq_len)
    K_T = [[K[j][i] for j in range(len(K))] for i in range(len(K[0]))]
    scores = matrix_multiply(Q, K_T)

    # Apply scale
    scores = [[s / scale for s in row] for row in scores]

    # Apply mask (additive: -inf for masked positions)
    if mask is not None:
        for i in range(seq_len):
            for j in range(len(scores[i])):
                if mask[i][j] == 0:
                    scores[i][j] -= 1e9

    # Softmax
    attn = softmax_rows(scores)

    # Output = attn * V: (seq_len x d_v)
    output = matrix_multiply(attn, V)
    return output


def _rand_matrix(rows, cols, seed, scale=0.01):
    """Generate a small random weight matrix."""
    rng = random.Random(seed)
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


class AttentionLayer:
    """Single-head scaled dot-product attention layer."""

    def __init__(self, d_model: int, d_key: int, seed: int = 0):
        self.d_model = d_model
        self.d_key = d_key
        # Weight matrices: W_Q, W_K, W_V: (d_model x d_key)
        self.W_Q = _rand_matrix(d_model, d_key, seed)
        self.W_K = _rand_matrix(d_model, d_key, seed + 1)
        self.W_V = _rand_matrix(d_model, d_key, seed + 2)

    def forward(self, X):
        """
        Forward pass.
        X: (seq_len x d_model)
        Returns: (seq_len x d_key)
        """
        Q = matrix_multiply(X, self.W_Q)
        K = matrix_multiply(X, self.W_K)
        V = matrix_multiply(X, self.W_V)
        return scaled_dot_product_attention(Q, K, V)


class AttentionAgent:
    """
    Multi-head attention agent for market time series.
    Maintains a sliding window buffer and produces action predictions.
    """

    def __init__(
        self,
        state_dim: int,
        seq_len: int,
        d_model: int = 32,
        n_heads: int = 2,
        action_dim: int = 3,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.action_dim = action_dim

        # Input projection: state_dim -> d_model
        self.W_in = _rand_matrix(state_dim, d_model, seed)

        # Attention layers (one per head)
        d_head = max(1, d_model // n_heads)
        self.heads = [AttentionLayer(d_model, d_head, seed=seed + i * 7) for i in range(n_heads)]

        # Output projection: (n_heads * d_head) -> action_dim
        concat_dim = n_heads * d_head
        self.W_out = _rand_matrix(concat_dim, action_dim, seed + 100)
        self.b_out = [0.0] * action_dim

        # Sliding window buffer
        self._buffer = []

    def _positional_encoding(self, seq_len: int, d_model: int) -> list:
        """Sinusoidal positional encoding. Returns (seq_len x d_model)."""
        pe = []
        for pos in range(seq_len):
            row = []
            for i in range(d_model):
                if i % 2 == 0:
                    angle = pos / (10000 ** (i / d_model))
                    row.append(math.sin(angle))
                else:
                    angle = pos / (10000 ** ((i - 1) / d_model))
                    row.append(math.cos(angle))
            pe.append(row)
        return pe

    def encode(self, state_sequence: list) -> list:
        """
        Encode a state sequence through positional encoding + attention layers.
        state_sequence: list of seq_len state vectors (each of length state_dim)
        Returns: encoded representation (seq_len x (n_heads * d_head))
        """
        seq_len = len(state_sequence)
        if seq_len == 0:
            return []

        # Project input: (seq_len x d_model)
        X = matrix_multiply(state_sequence, self.W_in)

        # Add positional encoding
        pe = self._positional_encoding(seq_len, self.d_model)
        X = [[X[i][j] + pe[i][j] for j in range(self.d_model)] for i in range(seq_len)]

        # Multi-head attention: collect outputs from each head
        head_outputs = [head.forward(X) for head in self.heads]

        # Concatenate head outputs: (seq_len x (n_heads * d_head))
        concat = []
        for i in range(seq_len):
            row = []
            for h_out in head_outputs:
                row.extend(h_out[i])
            concat.append(row)

        return concat

    def predict_action(self, state_sequence: list) -> int:
        """
        Predict action from state sequence.
        Returns action index (0=flat, 1=long, 2=short).
        """
        if len(state_sequence) == 0:
            return 0

        encoded = self.encode(state_sequence)
        if not encoded:
            return 0

        # Use last timestep's encoding
        last = encoded[-1]

        # Output projection: logits
        logits = [sum(last[j] * self.W_out[j][a] for j in range(len(last))) + self.b_out[a]
                  for a in range(self.action_dim)]

        # Argmax
        best_a = 0
        best_v = logits[0]
        for a in range(1, self.action_dim):
            if logits[a] > best_v:
                best_v = logits[a]
                best_a = a

        return best_a

    def update_sequence(self, new_state: list) -> None:
        """Update sliding window buffer with new state vector."""
        self._buffer.append(new_state)
        if len(self._buffer) > self.seq_len:
            self._buffer = self._buffer[-self.seq_len:]

    def predict_from_buffer(self) -> int:
        """Predict action using the internal sliding window buffer."""
        if len(self._buffer) == 0:
            return 0
        # Pad if needed
        seq = self._buffer[:]
        if len(seq) < self.seq_len:
            pad = [0.0] * self.state_dim
            seq = [pad] * (self.seq_len - len(seq)) + seq
        return self.predict_action(seq)

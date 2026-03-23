"""Pure Python LSTM agent for sequence prediction (no numpy/torch required)."""

from __future__ import annotations

import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Scalar activation functions
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)


# ---------------------------------------------------------------------------
# LCG-based Xavier initializer
# ---------------------------------------------------------------------------

def _lcg_xavier_init(rows: int, cols: int, seed: int) -> List[List[float]]:
    """
    Initialize a rows×cols weight matrix with Xavier uniform initialization
    using an LCG PRNG for reproducibility without external libraries.
    """
    limit = math.sqrt(6.0 / (rows + cols))
    state = seed & 0xFFFFFFFFFFFFFFFF
    weights: List[List[float]] = []
    for _ in range(rows):
        row: List[float] = []
        for _ in range(cols):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            # Map to [0, 1)
            f = (state >> 11) / (1 << 53)
            # Map to [-limit, limit]
            row.append(f * 2.0 * limit - limit)
        weights.append(row)
    return weights


def _zero_vec(n: int) -> List[float]:
    return [0.0] * n


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LstmCell:
    """Single LSTM cell (no external dependencies)."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        h = hidden_size
        i = input_size

        # Weight matrices: W[h × (h+i)] for each of the four gates (i, f, g, o)
        # Concatenated input [h_prev; x] has dimension h+i
        concat_size = h + i
        self.W_i = _lcg_xavier_init(h, concat_size, seed)
        self.W_f = _lcg_xavier_init(h, concat_size, seed + 1)
        self.W_g = _lcg_xavier_init(h, concat_size, seed + 2)
        self.W_o = _lcg_xavier_init(h, concat_size, seed + 3)

        # Bias vectors
        self.b_i = _zero_vec(h)
        self.b_f = [1.0] * h   # forget gate bias initialized to 1
        self.b_g = _zero_vec(h)
        self.b_o = _zero_vec(h)

    def mat_vec(self, W: List[List[float]], x: List[float]) -> List[float]:
        """Matrix-vector product W @ x."""
        result = []
        for row in W:
            s = 0.0
            for w, xi in zip(row, x):
                s += w * xi
            result.append(s)
        return result

    def vec_add(self, a: List[float], b: List[float]) -> List[float]:
        """Element-wise vector addition."""
        return [ai + bi for ai, bi in zip(a, b)]

    def vec_apply(self, a: List[float], f) -> List[float]:  # type: ignore[type-arg]
        """Apply scalar function element-wise."""
        return [f(ai) for ai in a]

    def forward(
        self,
        x: List[float],
        h: List[float],
        c: List[float],
    ) -> Tuple[List[float], List[float]]:
        """
        One LSTM step.

        Parameters
        ----------
        x : input vector (input_size,)
        h : previous hidden state (hidden_size,)
        c : previous cell state (hidden_size,)

        Returns
        -------
        (h_new, c_new)
        """
        # Concatenate [h; x]
        concat = h + x

        # Gate pre-activations
        i_pre = self.vec_add(self.mat_vec(self.W_i, concat), self.b_i)
        f_pre = self.vec_add(self.mat_vec(self.W_f, concat), self.b_f)
        g_pre = self.vec_add(self.mat_vec(self.W_g, concat), self.b_g)
        o_pre = self.vec_add(self.mat_vec(self.W_o, concat), self.b_o)

        # Gate activations
        i_gate = self.vec_apply(i_pre, sigmoid)
        f_gate = self.vec_apply(f_pre, sigmoid)
        g_gate = self.vec_apply(g_pre, tanh)
        o_gate = self.vec_apply(o_pre, sigmoid)

        # Cell state update: c_new = f * c + i * g
        c_new = [
            f_gate[j] * c[j] + i_gate[j] * g_gate[j]
            for j in range(self.hidden_size)
        ]

        # Hidden state: h_new = o * tanh(c_new)
        h_new = [o_gate[j] * tanh(c_new[j]) for j in range(self.hidden_size)]

        return h_new, c_new


# ---------------------------------------------------------------------------
# LSTM Agent (stacked LSTM + linear output)
# ---------------------------------------------------------------------------

class LstmAgent:
    """
    Stacked LSTM agent for sequence-based action prediction.

    Architecture: num_layers LstmCells → linear output head.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        action_dim: int,
        num_layers: int = 2,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.num_layers = num_layers

        # Build stacked LSTM cells
        self.cells: List[LstmCell] = []
        for layer in range(num_layers):
            in_size = state_dim if layer == 0 else hidden_size
            self.cells.append(LstmCell(in_size, hidden_size, seed=seed + layer * 100))

        # Output linear layer: hidden_size → action_dim
        self.W_out = _lcg_xavier_init(action_dim, hidden_size, seed + num_layers * 100)
        self.b_out = _zero_vec(action_dim)

        # Hidden/cell states
        self.reset_hidden()

    def reset_hidden(self) -> None:
        """Reset all hidden and cell states to zero."""
        self._h: List[List[float]] = [_zero_vec(self.hidden_size) for _ in range(self.num_layers)]
        self._c: List[List[float]] = [_zero_vec(self.hidden_size) for _ in range(self.num_layers)]

    def update_hidden(self, state: List[float]) -> None:
        """Advance the hidden state by one step."""
        x = state
        for layer in range(self.num_layers):
            h_new, c_new = self.cells[layer].forward(x, self._h[layer], self._c[layer])
            self._h[layer] = h_new
            self._c[layer] = c_new
            x = h_new

    def linear(
        self, x: List[float], W: List[List[float]], b: List[float]
    ) -> List[float]:
        """Linear transform: W @ x + b."""
        cell = self.cells[0]
        out = cell.mat_vec(W, x)
        return [out[j] + b[j] for j in range(len(b))]

    def softmax(self, x: List[float]) -> List[float]:
        """Numerically stable softmax."""
        max_x = max(x) if x else 0.0
        exps = [math.exp(xi - max_x) for xi in x]
        total = sum(exps)
        return [e / total for e in exps]

    def forward(self, state_sequence: List[List[float]]) -> List[float]:
        """
        Run the full sequence through the LSTM and return action logits.

        Parameters
        ----------
        state_sequence : list of state vectors (sequence_length × state_dim)

        Returns
        -------
        action logits (action_dim,)
        """
        # Save current hidden state, process sequence, restore
        h_saved = [list(h) for h in self._h]
        c_saved = [list(c) for c in self._c]

        self.reset_hidden()
        for state in state_sequence:
            self.update_hidden(state)

        logits = self.linear(self._h[-1], self.W_out, self.b_out)

        self._h = h_saved
        self._c = c_saved
        return logits

    def predict_action(self, state: List[float]) -> int:
        """Predict action for a single state (advances hidden state)."""
        self.update_hidden(state)
        logits = self.linear(self._h[-1], self.W_out, self.b_out)
        probs = self.softmax(logits)
        return probs.index(max(probs))

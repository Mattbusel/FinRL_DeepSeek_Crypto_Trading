"""Neural network architectures for the LARSA RNN factor-mining pipeline.

Provides :class:`NnSeqBnMLP` (a sequence-aware MLP) and
:class:`RnnRegNet` (a stacked LSTM + GRU regression network) used by
:mod:`seq_run` to predict short-term price-movement labels from
Alpha101 + sentiment features.

Example::

    from seq_net import RnnRegNet
    net = RnnRegNet(inp_dim=20, mid_dim=128, out_dim=8, num_layers=4)
"""

from __future__ import annotations

import torch as th
import torch.nn as nn

TEN = th.Tensor


def layer_init_with_orthogonal(
    layer: nn.Linear, std: float = 1.0, bias_const: float = 1e-6
) -> None:
    """Initialise a linear layer with orthogonal weights.

    Args:
        layer: The :class:`torch.nn.Linear` layer to initialise.
        std: Scaling factor for the orthogonal weight matrix.
        bias_const: Constant value to fill the bias term with.
    """
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class NnSeqBnMLP(nn.Module):
    """Sequence-aware MLP that reshapes ``(T, B, F)`` tensors for linear layers.

    Applies an optional input BatchNorm, followed by a stack of Linear +
    GELU + optional LayerNorm layers.  All linear operations are performed
    on the flattened ``(T*B, F)`` view and the output is reshaped back to
    ``(T, B, out_dim)``.

    Args:
        dims: Layer widths ``[in_dim, hidden_dim, ..., out_dim]``.
        if_inp_norm: Whether to prepend a :class:`torch.nn.BatchNorm1d` on
            the input features.
        if_layer_norm: Whether to insert a :class:`torch.nn.LayerNorm` after
            each hidden GELU activation.
        activation: Optional activation module appended after the last linear
            layer (e.g. ``nn.Tanh()``).
    """

    def __init__(
        self,
        dims: list[int],
        if_inp_norm: bool = False,
        if_layer_norm: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        mlp_list: list[nn.Module] = []
        if if_inp_norm:
            mlp_list.append(nn.BatchNorm1d(dims[0], momentum=0.9))

        mlp_list.append(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims) - 1):
            mlp_list.append(nn.GELU())
            if if_layer_norm:
                mlp_list.append(nn.LayerNorm(dims[i]))
            mlp_list.append(nn.Linear(dims[i], dims[i + 1]))

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)  # type: ignore[arg-type]
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)  # type: ignore[arg-type]

    def forward(self, seq: TEN) -> TEN:
        """Apply the MLP to a 3-D sequence tensor.

        Args:
            seq: Input tensor of shape ``(T, B, in_dim)``.

        Returns:
            Output tensor of shape ``(T, B, out_dim)``.
        """
        d0, d1, d2 = seq.shape
        inp = seq.reshape(d0 * d1, -1)
        out = self.mlp(inp)
        return out.reshape(d0, d1, -1)

    def reset_parameters(self, std: float = 1.0, bias_const: float = 1e-6) -> None:
        """Re-initialise all linear and batch-norm parameters.

        Args:
            std: Orthogonal weight scale for linear layers; constant weight
                for BatchNorm.
            bias_const: Constant bias initialisation value.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                th.nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_const)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, std)
                nn.init.constant_(module.bias, 0)


class RnnRegNet(nn.Module):
    """Stacked LSTM + GRU regression network for sequence-to-sequence prediction.

    Architecture:
    1. Input MLP (``inp_dim -> mid_dim``) with GELU activation.
    2. LSTM (``mid_dim -> mid_dim``, ``num_layers`` stacked) + residual MLP.
    3. GRU  (``mid_dim -> mid_dim``, ``num_layers`` stacked) + residual MLP.
    4. Output MLP (``mid_dim*2 -> out_dim``) with Tanh activation.

    Args:
        inp_dim: Dimensionality of each input feature vector.
        mid_dim: Width of all hidden layers.
        out_dim: Dimensionality of each output prediction vector.
        num_layers: Number of stacked recurrent layers in both LSTM and GRU.
    """

    def __init__(
        self,
        inp_dim: int,
        mid_dim: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.rnn1 = nn.LSTM(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = NnSeqBnMLP(
            dims=[mid_dim, mid_dim, mid_dim], if_layer_norm=False, activation=nn.GELU()
        )

        self.rnn2 = nn.GRU(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp2 = NnSeqBnMLP(
            dims=[mid_dim, mid_dim, mid_dim], if_layer_norm=False, activation=nn.GELU()
        )

        self.mlp_inp = NnSeqBnMLP(
            dims=[inp_dim, mid_dim, mid_dim], if_layer_norm=False, activation=nn.GELU()
        )
        self.mlp_out = NnSeqBnMLP(
            dims=[mid_dim * 2, mid_dim * 2, out_dim],
            if_layer_norm=False,
            activation=nn.Tanh(),
        )

    def forward(
        self,
        inp: TEN,
        hid: tuple[TEN | None, TEN | None] | None = None,
    ) -> tuple[TEN, tuple[TEN | None, TEN | None]]:
        """Run a forward pass over the input sequence.

        Args:
            inp: Input tensor of shape ``(T, B, inp_dim)``.
            hid: Optional tuple ``(hid1, hid2)`` of hidden states for the
                LSTM and GRU respectively.  Pass ``None`` to zero-initialise.

        Returns:
            A tuple ``(out, (hid1, hid2))`` where *out* has shape
            ``(T, B, out_dim)`` and *hid1*, *hid2* are the updated hidden
            states.
        """
        hid1, hid2 = (None, None) if hid is None else hid
        inp = self.mlp_inp(inp)

        rnn1, hid1 = self.rnn1(inp, hid1)
        tmp1 = self.mlp1(rnn1)

        rnn2, hid2 = self.rnn2(inp, hid2)
        tmp2 = self.mlp2(rnn2)

        tmp = th.concat((tmp1, tmp2), dim=2)
        out = self.mlp_out(tmp)
        return out, (hid1, hid2)

    @staticmethod
    def get_obj_value(
        criterion: nn.Module,
        out: TEN,
        lab: TEN,
        wup_dim: int,
    ) -> TEN:
        """Compute the per-element MSE loss, excluding the warm-up prefix.

        Args:
            criterion: Loss function (e.g. :class:`torch.nn.MSELoss` with
                ``reduction='none'``).
            out: Network output tensor of shape ``(T, B, out_dim)``.
            lab: Target label tensor of same shape as *out*.
            wup_dim: Number of leading time steps to exclude (warm-up period).

        Returns:
            Loss tensor of shape ``(T - wup_dim, B, out_dim)``.
        """
        return criterion(out, lab)[wup_dim:, :, :]

"""Q-network architectures for ElegantRL DQN agents.

Provides :class:`QNetBase`, :class:`QNetTwin` (Double DQN), and
:class:`QNetTwinDuel` (Dueling Double DQN / D3QN), along with helper
utilities :func:`build_mlp` and :func:`layer_init_with_orthogonal`.

Example::

    from erl_net import QNetTwinDuel
    net = QNetTwinDuel(dims=[128, 128, 128], state_dim=12, action_dim=3)
"""

from __future__ import annotations

import torch
import torch.nn as nn

TEN = torch.Tensor


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    """Abstract base class for all Q-network variants.

    Provides shared state and value normalisation parameters.

    Args:
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        """Normalise a state tensor using the running mean and std.

        Args:
            state: Raw observation tensor.

        Returns:
            Normalised tensor ``(state - state_avg) / state_std``.
        """
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: TEN) -> TEN:
        """Re-normalise a Q-value tensor back to the original scale.

        Args:
            value: Normalised Q-value tensor.

        Returns:
            Tensor ``value * value_std + value_avg``.
        """
        return value * self.value_std + self.value_avg


class QNetTwin(QNetBase):  # Double DQN
    """Twin-head Q-network for Double DQN.

    Two independent value heads share a common state encoder, providing the
    two Q-value estimates required by the Double DQN update.

    Args:
        dims: Hidden-layer widths of the shared state encoder MLP.
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
    """

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        """Compute Q-values from the first value head.

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Q-value tensor of shape ``(batch, action_dim)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        """Compute re-normalised Q-values from both value heads.

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Tuple ``(q_val1, q_val2)`` each of shape ``(batch, action_dim)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        """Select an action using epsilon-greedy exploration.

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Action index tensor of shape ``(batch, 1)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = torch.multinomial(a_prob, num_samples=1)
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN
    """Dueling Double DQN network (D3QN).

    Uses the dueling architecture — separate advantage and value streams —
    combined with twin heads for Double DQN variance reduction.

    Args:
        dims: Hidden-layer widths of the shared state encoder MLP.
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
    """

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        """Compute dueling Q-values from the first head.

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Dueling Q-value tensor of shape ``(batch, action_dim)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state):
        """Compute re-normalised dueling Q-values from both heads.

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Tuple ``(q_duel1, q_duel2)`` each of shape ``(batch, action_dim)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1
        q_duel1 = self.value_re_norm(q_duel1)

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        q_duel2 = self.value_re_norm(q_duel2)
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        """Select an action using epsilon-greedy exploration (dueling).

        Args:
            state: Observation tensor of shape ``(batch, state_dim)``.

        Returns:
            Action index tensor of shape ``(batch, 1)``.
        """
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = torch.multinomial(a_prob, num_samples=1)
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """Build a Multi-Layer Perceptron as a :class:`torch.nn.Sequential`.

    Args:
        dims: List of layer widths; ``dims[0]`` is the input width and
            ``dims[-1]`` is the output width.
        activation: Activation class to insert between layers (default:
            :class:`torch.nn.ReLU`).
        if_raw_out: When ``True`` (default), removes the final activation so
            the output is a raw (unbounded) linear projection.

    Returns:
        A :class:`torch.nn.Sequential` containing alternating linear and
        activation layers.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    """Initialise a linear layer with orthogonal weights.

    Args:
        layer: A :class:`torch.nn.Linear` layer to initialise in-place.
        std: Gain passed to :func:`torch.nn.init.orthogonal_` (default 1.0).
        bias_const: Constant value for bias initialisation (default 1e-6).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

"""Federated learning scaffold for decentralised RL agent training.

Simulates a federated setup where multiple :class:`FederatedClient` instances
(representing different exchanges or strategies) train locally on private data
and send gradient updates to a :class:`FederatedServer`, which aggregates them
via FedAvg.  Differential privacy (DP) noise is added to each client's
gradient before transmission to provide a formal privacy guarantee.

Key components
--------------
* :class:`FederatedClient` -- local trainer; holds private data and model copy.
* :class:`FederatedServer` -- aggregates client updates via weighted FedAvg.
* :func:`add_dp_noise` -- adds calibrated Gaussian noise to gradients.
* :class:`FederatedRound` -- orchestrates one round of FL (broadcast → train
  → aggregate).

Privacy guarantee
-----------------
Each gradient tensor is clipped to ``max_grad_norm`` and then Gaussian noise
N(0, (noise_multiplier * max_grad_norm)^2) is added before upload.  With
enough clients and appropriate noise_multiplier, this provides (epsilon,
delta)-DP.

Example::

    from federated_learning import FederatedServer, FederatedClient, FederatedRound

    server = FederatedServer(global_model)
    clients = [FederatedClient(model_copy, local_data_i) for i in range(5)]
    for round_idx in range(10):
        fed_round = FederatedRound(server, clients)
        metrics = fed_round.run(local_steps=100)
        print(metrics)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Differential privacy helpers
# ---------------------------------------------------------------------------


def clip_gradients(model: Any, max_norm: float) -> None:
    """Clip per-parameter gradients to ``max_norm`` in-place.

    Args:
        model: PyTorch ``nn.Module`` whose ``.parameters()`` will be clipped.
        max_norm: L2 norm clip value.
    """
    if not _TORCH_AVAILABLE:
        return
    import torch  # noqa: F811
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def add_dp_noise(
    model: Any,
    noise_multiplier: float,
    max_grad_norm: float,
    device: str = "cpu",
) -> None:
    """Add Gaussian DP noise to model gradients in-place.

    After clipping to ``max_grad_norm``, each gradient tensor receives
    additive Gaussian noise with std = ``noise_multiplier * max_grad_norm``.

    Args:
        model: PyTorch ``nn.Module``.
        noise_multiplier: Noise scale relative to the clipping norm.
        max_grad_norm: L2 norm used for gradient clipping.
        device: Device string for the noise tensor.
    """
    if not _TORCH_AVAILABLE:
        return
    import torch  # noqa: F811

    sigma = noise_multiplier * max_grad_norm
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad, device=device) * sigma
            param.grad.data.add_(noise)


def extract_gradients(model: Any) -> List["np.ndarray"]:
    """Extract current gradients as a list of numpy arrays.

    Args:
        model: PyTorch ``nn.Module``.

    Returns:
        List of numpy arrays, one per parameter with a gradient.
    """
    if not _TORCH_AVAILABLE:
        return []
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().cpu().numpy().copy())
        else:
            grads.append(np.zeros(param.shape))
    return grads


def apply_gradients(model: Any, grads: List["np.ndarray"], lr: float = 1.0) -> None:
    """Apply a list of gradient arrays to model parameters (SGD-style).

    Args:
        model: PyTorch ``nn.Module``.
        grads: Gradient arrays in the same order as ``model.parameters()``.
        lr: Learning rate scaling factor.
    """
    if not _TORCH_AVAILABLE:
        return
    import torch  # noqa: F811
    for param, g in zip(model.parameters(), grads):
        param.data.sub_(lr * torch.tensor(g, device=param.device, dtype=param.dtype))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClientUpdate:
    """Gradient update package sent from a client to the server.

    Attributes:
        client_id: Unique identifier for the originating client.
        gradients: List of gradient arrays (one per model parameter).
        num_samples: Number of local training samples used (for FedAvg weighting).
        loss: Local training loss for logging.
    """

    client_id: str
    gradients: List["np.ndarray"]
    num_samples: int
    loss: float


@dataclass
class RoundMetrics:
    """Metrics produced after one federated round.

    Attributes:
        round_idx: Round number (0-based).
        num_clients: Number of clients that participated.
        mean_client_loss: Average local training loss across clients.
        aggregated: True when the server successfully aggregated updates.
    """

    round_idx: int
    num_clients: int
    mean_client_loss: float
    aggregated: bool


# ---------------------------------------------------------------------------
# DP config
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Differential privacy configuration.

    Attributes:
        enabled: Whether to apply DP noise.
        noise_multiplier: Gaussian noise scale (relative to max_grad_norm).
        max_grad_norm: L2 gradient clipping threshold.
    """

    enabled: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Federated client
# ---------------------------------------------------------------------------


class FederatedClient:
    """Federated learning client that trains on private local data.

    Args:
        model: A local copy of the global model (``nn.Module``).
        local_data: Tuple ``(X, y)`` of local training tensors or arrays.
        client_id: Unique string identifier.
        dp_config: Differential privacy settings.
        device: Torch device string.
        lr: Local learning rate.
    """

    def __init__(
        self,
        model: Any,
        local_data: Tuple[Any, Any],
        client_id: str = "client_0",
        dp_config: Optional[DPConfig] = None,
        device: str = "cpu",
        lr: float = 1e-3,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.dp_config = dp_config or DPConfig()
        self.device = device
        self.lr = lr

        if _TORCH_AVAILABLE:
            import torch  # noqa: F811
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_weights(self, state_dict: Dict[str, Any]) -> None:
        """Update local model weights from the global server state dict.

        Args:
            state_dict: PyTorch state dictionary from the global model.
        """
        if not _TORCH_AVAILABLE:
            return
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def train_local(
        self,
        num_steps: int = 50,
        loss_fn: Optional[Any] = None,
    ) -> ClientUpdate:
        """Train locally for *num_steps* gradient steps and return update.

        Args:
            num_steps: Number of local optimisation steps.
            loss_fn: Loss function ``(output, target) -> scalar``.  Defaults
                to MSE when not supplied.

        Returns:
            :class:`ClientUpdate` with (optionally noisy) gradients ready
            for the server.
        """
        if not _TORCH_AVAILABLE:
            return ClientUpdate(
                client_id=self.client_id,
                gradients=[],
                num_samples=0,
                loss=0.0,
            )

        import torch  # noqa: F811
        import torch.nn.functional as F

        if loss_fn is None:
            loss_fn = F.mse_loss

        X, y = self.local_data
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        num_samples = len(X)
        total_loss = 0.0

        self.model.train()
        for _ in range(num_steps):
            # Mini-batch sampling (full batch for simplicity)
            output = self.model(X)
            loss = loss_fn(output.squeeze(-1), y)

            self._optimizer.zero_grad()
            loss.backward()

            if self.dp_config.enabled:
                clip_gradients(self.model, self.dp_config.max_grad_norm)
                add_dp_noise(
                    self.model,
                    self.dp_config.noise_multiplier,
                    self.dp_config.max_grad_norm,
                    device=self.device,
                )

            self._optimizer.step()
            total_loss += float(loss.item())

        self.model.eval()
        avg_loss = total_loss / max(num_steps, 1)
        grads = extract_gradients(self.model)

        log.debug(
            "federated_client.local_train",
            client_id=self.client_id,
            num_steps=num_steps,
            avg_loss=round(avg_loss, 6),
            dp_enabled=self.dp_config.enabled,
        )
        return ClientUpdate(
            client_id=self.client_id,
            gradients=grads,
            num_samples=num_samples,
            loss=avg_loss,
        )


# ---------------------------------------------------------------------------
# Federated server
# ---------------------------------------------------------------------------


class FederatedServer:
    """Aggregates client gradient updates via FedAvg.

    Args:
        global_model: The authoritative global model (``nn.Module``).
        server_lr: Server-side learning rate applied to the aggregated gradient.
    """

    def __init__(self, global_model: Any, server_lr: float = 1.0) -> None:
        self.global_model = global_model
        self.server_lr = server_lr
        self._round = 0

    def get_global_weights(self) -> Dict[str, Any]:
        """Return a deep copy of the global model's state dictionary."""
        if not _TORCH_AVAILABLE:
            return {}
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, updates: List[ClientUpdate]) -> bool:
        """FedAvg aggregation: weighted average of client gradients.

        Weights are proportional to ``num_samples`` of each client.

        Args:
            updates: List of :class:`ClientUpdate` from participating clients.

        Returns:
            True on success, False when no valid updates were supplied.
        """
        if not updates or not _TORCH_AVAILABLE:
            return False

        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            return False

        # Weighted average of gradient arrays
        num_params = len(updates[0].gradients)
        avg_grads: List["np.ndarray"] = []

        for p_idx in range(num_params):
            weighted = sum(
                (u.num_samples / total_samples) * u.gradients[p_idx]
                for u in updates
                if len(u.gradients) > p_idx
            )
            avg_grads.append(weighted)

        apply_gradients(self.global_model, avg_grads, lr=self.server_lr)
        self._round += 1
        log.info(
            "federated_server.aggregate",
            round=self._round,
            num_clients=len(updates),
            total_samples=total_samples,
        )
        return True


# ---------------------------------------------------------------------------
# Federated round orchestrator
# ---------------------------------------------------------------------------


class FederatedRound:
    """Orchestrates one complete federated learning round.

    Steps:
    1. Broadcast global weights to all clients.
    2. Each client trains locally and returns gradients.
    3. Server aggregates via FedAvg.

    Args:
        server: :class:`FederatedServer` instance.
        clients: List of :class:`FederatedClient` instances.
    """

    def __init__(
        self,
        server: FederatedServer,
        clients: List[FederatedClient],
    ) -> None:
        self.server = server
        self.clients = clients

    def run(
        self,
        round_idx: int = 0,
        local_steps: int = 50,
        loss_fn: Optional[Any] = None,
        client_fraction: float = 1.0,
    ) -> RoundMetrics:
        """Execute one federated round.

        Args:
            round_idx: Round index for logging.
            local_steps: Steps each client trains locally.
            loss_fn: Loss function forwarded to each client.
            client_fraction: Fraction of clients to sample (default: all).

        Returns:
            :class:`RoundMetrics` summarising the round.
        """
        # 1. Broadcast global weights
        global_weights = self.server.get_global_weights()
        selected = self._sample_clients(client_fraction)
        for client in selected:
            client.set_weights(global_weights)

        # 2. Local training
        updates: List[ClientUpdate] = []
        for client in selected:
            update = client.train_local(num_steps=local_steps, loss_fn=loss_fn)
            updates.append(update)

        # 3. Aggregate
        ok = self.server.aggregate(updates)

        mean_loss = float(np.mean([u.loss for u in updates])) if updates else 0.0
        metrics = RoundMetrics(
            round_idx=round_idx,
            num_clients=len(selected),
            mean_client_loss=mean_loss,
            aggregated=ok,
        )
        log.info(
            "federated_round.complete",
            round_idx=round_idx,
            num_clients=len(selected),
            mean_loss=round(mean_loss, 6),
            aggregated=ok,
        )
        return metrics

    def _sample_clients(self, fraction: float) -> List[FederatedClient]:
        if fraction >= 1.0:
            return self.clients
        k = max(1, math.ceil(len(self.clients) * fraction))
        indices = np.random.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in indices]

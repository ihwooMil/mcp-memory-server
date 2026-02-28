"""Byzantine-tolerant gossip protocol for distributed policy learning.

Provides:
- Transport (ABC): abstract transport layer for peer communication
- InMemoryTransport: in-process transport for testing
- GossipNode: peer node with Krum aggregation and norm clipping
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from aimemory.online.policy import OnlinePolicy

if TYPE_CHECKING:
    from aimemory.online.rule_verifier import RuleVerifier

logger = logging.getLogger(__name__)


# ─── Transport abstraction ────────────────────────────────────────────


class Transport(ABC):
    """Abstract transport layer for gossip communication."""

    @abstractmethod
    def send(self, peer_id: str, data: np.ndarray) -> None:
        """Send a parameter delta to a peer."""

    @abstractmethod
    def receive(self) -> list[tuple[str, np.ndarray]]:
        """Receive all pending updates. Returns list of (peer_id, delta)."""


class InMemoryTransport(Transport):
    """In-process transport for testing. All nodes share a message bus."""

    def __init__(
        self,
        node_id: str,
        bus: dict[str, list[tuple[str, np.ndarray]]],
        rule_hash_bus: dict[str, list[tuple[str, bytes]]] | None = None,
    ) -> None:
        self._node_id = node_id
        self._bus = bus
        self._rule_hash_bus: dict[str, list[tuple[str, bytes]]] = (
            rule_hash_bus if rule_hash_bus is not None else {}
        )
        # Ensure our inbox exists
        if node_id not in self._bus:
            self._bus[node_id] = []

    def send(self, peer_id: str, data: np.ndarray) -> None:
        if peer_id not in self._bus:
            self._bus[peer_id] = []
        self._bus[peer_id].append((self._node_id, data.copy()))

    def receive(self) -> list[tuple[str, np.ndarray]]:
        messages = self._bus.get(self._node_id, [])
        self._bus[self._node_id] = []
        return messages

    def send_rule_hash(self, peer_id: str, rule_hash: bytes) -> None:
        """Send a rule hash to a peer."""
        if peer_id not in self._rule_hash_bus:
            self._rule_hash_bus[peer_id] = []
        self._rule_hash_bus[peer_id].append((self._node_id, rule_hash))

    def receive_rule_hashes(self) -> list[tuple[str, bytes]]:
        """Drain and return all pending rule hash messages."""
        hashes = list(self._rule_hash_bus.get(self._node_id, []))
        self._rule_hash_bus[self._node_id] = []
        return hashes


# ─── Krum aggregation ─────────────────────────────────────────────────


def krum(updates: list[np.ndarray], f: int) -> np.ndarray:
    """Select the update closest to the majority using Krum algorithm.

    Given n updates and at most f Byzantine peers (f < (n-3)/2),
    Krum selects the update whose sum of distances to its (n-f-2) nearest
    neighbors is minimal.

    Args:
        updates: List of parameter delta vectors.
        f: Maximum number of Byzantine (adversarial) peers to tolerate.

    Returns:
        The selected safe update vector.

    Raises:
        ValueError: If fewer than 3 updates or f is too large.
    """
    n = len(updates)
    if n < 1:
        raise ValueError("Need at least 1 update for Krum")
    if n == 1:
        return updates[0]
    if n == 2:
        # With 2 updates, can't filter; return the mean
        return (updates[0] + updates[1]) / 2.0

    # f must satisfy: n - f - 2 >= 1, i.e., f <= n - 3
    f = min(f, n - 3)
    if f < 0:
        f = 0
    k = n - f - 2  # number of nearest neighbors to consider

    if k < 1:
        k = 1

    # Compute pairwise squared distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((updates[i] - updates[j]) ** 2))
            distances[i][j] = d
            distances[j][i] = d

    # For each update, sum distances to its k nearest neighbors
    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        # sorted_dists[0] is always 0 (self), take next k
        scores[i] = np.sum(sorted_dists[1 : 1 + k])

    best_idx = int(np.argmin(scores))
    return updates[best_idx]


# ─── GossipNode ───────────────────────────────────────────────────────


class GossipNode:
    """A gossip node that exchanges policy parameters with peers.

    Features:
    - Norm clipping: outgoing deltas are L2-clipped to max_norm
    - Differential privacy: optional Gaussian noise added after clipping
    - Krum aggregation: incoming updates are filtered using Krum algorithm
    - Rule hash verification: optional peer integrity checking
    - Periodic gossip: step() triggers send/aggregate at configured interval
    """

    def __init__(
        self,
        node_id: str,
        policy: OnlinePolicy,
        transport: Transport,
        max_norm: float = 1.0,
        gossip_interval: int = 50,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_enabled: bool = False,
        rule_verifier: RuleVerifier | None = None,
    ) -> None:
        self.node_id = node_id
        self._policy = policy
        self._transport = transport
        self._max_norm = max_norm
        self._gossip_interval = gossip_interval
        self._dp_epsilon = dp_epsilon
        self._dp_delta = dp_delta
        self._dp_enabled = dp_enabled
        self._rule_verifier = rule_verifier
        self._rejected_peers: set[str] = set()
        self._peers: dict[str, Transport] = {}
        self._receive_buffer: list[tuple[str, np.ndarray]] = []
        self._step_count: int = 0
        self._base_params: np.ndarray = policy.get_parameters().copy()

    def register_peer(self, peer_id: str, transport: Transport) -> None:
        """Register a peer for gossip communication."""
        self._peers[peer_id] = transport

    def send_update(self) -> None:
        """Compute delta from base, norm-clip it, optionally add DP noise, and send to all peers."""
        current_params = self._policy.get_parameters()
        delta = current_params - self._base_params

        # L2 norm clipping
        delta = _clip_norm(delta, self._max_norm)

        # Differential privacy: add calibrated Gaussian noise
        if self._dp_enabled:
            delta = _add_dp_noise(delta, self._max_norm, self._dp_epsilon, self._dp_delta)

        for peer_id in self._peers:
            self._transport.send(peer_id, delta)

        # Update base to current
        self._base_params = current_params.copy()

    def send_rule_hash(self) -> None:
        """Send our rule hash to all peers (call before send_update)."""
        if not self._rule_verifier:
            return
        if not hasattr(self._transport, "send_rule_hash"):
            return
        for peer_id in self._peers:
            self._transport.send_rule_hash(peer_id, self._rule_verifier.rule_hash_bytes)

    def receive_update(self, peer_id: str, delta: np.ndarray) -> None:
        """Store a received delta in the buffer."""
        self._receive_buffer.append((peer_id, delta))

    def aggregate(self) -> None:
        """Aggregate buffered updates using Krum and apply to policy."""
        # Drain transport inbox into buffer
        inbox = self._transport.receive()

        # Rule hash verification (if verifier and transport support it)
        if self._rule_verifier and hasattr(self._transport, "receive_rule_hashes"):
            for peer_id, hash_bytes in self._transport.receive_rule_hashes():
                if not self._rule_verifier.verify(hash_bytes):
                    logger.warning("Rule hash mismatch from peer %s — rejecting", peer_id)
                    self._rejected_peers.add(peer_id)

        for peer_id, delta in inbox:
            if peer_id in self._rejected_peers:
                logger.info("Dropping delta from rejected peer %s", peer_id)
                continue
            self._receive_buffer.append((peer_id, delta))

        if not self._receive_buffer:
            return

        deltas = [delta for _, delta in self._receive_buffer]
        n = len(deltas)
        f = max(0, math.floor((n - 3) / 2)) if n >= 3 else 0

        safe_delta = krum(deltas, f)

        # Apply the safe delta to current policy parameters
        current_params = self._policy.get_parameters()
        updated_params = current_params + safe_delta
        self._policy.set_parameters(updated_params)
        self._base_params = updated_params.copy()

        self._receive_buffer.clear()

    def step(self) -> None:
        """Periodic gossip step: send and aggregate at configured interval."""
        self._step_count += 1
        if self._step_count % self._gossip_interval == 0:
            self.send_update()
            self.aggregate()


def _clip_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector to have L2 norm at most max_norm."""
    norm = float(np.linalg.norm(delta))
    if norm > max_norm:
        delta = delta * (max_norm / norm)
    return delta


def _add_dp_noise(
    delta: np.ndarray,
    max_norm: float,
    epsilon: float,
    delta_dp: float,
) -> np.ndarray:
    """Add calibrated Gaussian noise for (epsilon, delta)-DP.

    Sensitivity is bounded by max_norm (from prior L2 clipping).
    sigma = max_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    sigma = max_norm * math.sqrt(2.0 * math.log(1.25 / delta_dp)) / epsilon
    noise = np.random.default_rng().normal(0.0, sigma, delta.shape)
    return delta + noise

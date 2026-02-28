"""Tests for Byzantine-tolerant gossip module."""

from __future__ import annotations

import numpy as np
import pytest

from aimemory.config import SecurityConfig
from aimemory.online.gossip import (
    GossipNode,
    InMemoryTransport,
    _add_dp_noise,
    _clip_norm,
    krum,
)
from aimemory.online.policy import FEATURE_DIM, OnlinePolicy
from aimemory.online.rule_verifier import RuleVerifier

# ─── Krum algorithm tests ────────────────────────────────────────────


class TestKrum:
    def test_single_update(self):
        update = np.array([1.0, 2.0, 3.0])
        result = krum([update], f=0)
        np.testing.assert_array_equal(result, update)

    def test_two_updates_returns_mean(self):
        u1 = np.array([1.0, 0.0])
        u2 = np.array([3.0, 0.0])
        result = krum([u1, u2], f=0)
        np.testing.assert_allclose(result, [2.0, 0.0])

    def test_majority_selection(self):
        """Krum should select the update closest to the honest majority."""
        honest = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.1, 0.9, 1.0]),
            np.array([0.9, 1.1, 1.0]),
        ]
        adversarial = [np.array([100.0, 100.0, 100.0])]
        all_updates = honest + adversarial

        result = krum(all_updates, f=1)
        distances = [float(np.linalg.norm(result - h)) for h in honest]
        assert min(distances) < 0.01

    def test_filters_adversarial_updates(self):
        rng = np.random.default_rng(42)
        dim = 10
        honest = [rng.normal(0, 0.1, dim) for _ in range(5)]
        adversarial = [rng.normal(50, 1, dim) for _ in range(2)]
        all_updates = honest + adversarial

        result = krum(all_updates, f=2)
        assert float(np.linalg.norm(result)) < 2.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            krum([], f=0)

    def test_identical_updates(self):
        u = np.array([1.0, 2.0, 3.0])
        result = krum([u.copy() for _ in range(5)], f=1)
        np.testing.assert_allclose(result, u)


# ─── Norm clipping tests ─────────────────────────────────────────────


class TestNormClipping:
    def test_within_norm_unchanged(self):
        delta = np.array([0.3, 0.4])
        clipped = _clip_norm(delta, max_norm=1.0)
        np.testing.assert_array_equal(clipped, delta)

    def test_exceeding_norm_clipped(self):
        delta = np.array([3.0, 4.0])
        clipped = _clip_norm(delta, max_norm=1.0)
        assert float(np.linalg.norm(clipped)) == pytest.approx(1.0, abs=1e-6)
        direction = delta / np.linalg.norm(delta)
        clipped_dir = clipped / np.linalg.norm(clipped)
        np.testing.assert_allclose(clipped_dir, direction, atol=1e-6)

    def test_zero_vector_unchanged(self):
        delta = np.zeros(5)
        clipped = _clip_norm(delta, max_norm=1.0)
        np.testing.assert_array_equal(clipped, delta)

    def test_exact_norm_unchanged(self):
        delta = np.array([0.6, 0.8])
        clipped = _clip_norm(delta, max_norm=1.0)
        np.testing.assert_allclose(clipped, delta, atol=1e-6)


# ─── InMemoryTransport tests ─────────────────────────────────────────


class TestInMemoryTransport:
    def test_send_receive(self):
        bus: dict[str, list] = {}
        t_a = InMemoryTransport("A", bus)
        t_b = InMemoryTransport("B", bus)

        data = np.array([1.0, 2.0, 3.0])
        t_a.send("B", data)

        messages = t_b.receive()
        assert len(messages) == 1
        peer_id, received = messages[0]
        assert peer_id == "A"
        np.testing.assert_array_equal(received, data)

    def test_receive_clears_inbox(self):
        bus: dict[str, list] = {}
        t_a = InMemoryTransport("A", bus)
        t_b = InMemoryTransport("B", bus)

        t_a.send("B", np.array([1.0]))
        t_b.receive()
        assert len(t_b.receive()) == 0

    def test_multiple_messages(self):
        bus: dict[str, list] = {}
        t_a = InMemoryTransport("A", bus)
        t_b = InMemoryTransport("B", bus)
        t_c = InMemoryTransport("C", bus)

        t_a.send("C", np.array([1.0]))
        t_b.send("C", np.array([2.0]))

        messages = t_c.receive()
        assert len(messages) == 2


# ─── GossipNode tests ────────────────────────────────────────────────


# Shared policy to avoid repeated torch init overhead
_shared_policy: OnlinePolicy | None = None


def _get_fresh_policy() -> OnlinePolicy:
    """Create a fresh policy by cloning parameters from a shared base."""
    global _shared_policy
    if _shared_policy is None:
        _shared_policy = OnlinePolicy(feature_dim=FEATURE_DIM, epsilon=0.0)
    policy = OnlinePolicy.__new__(OnlinePolicy)
    policy.feature_dim = _shared_policy.feature_dim
    policy.n_actions = _shared_policy.n_actions
    policy.epsilon = _shared_policy.epsilon
    policy._rng = np.random.default_rng()

    import torch

    from aimemory.online.policy import _BanditMLP

    policy._model = _BanditMLP(FEATURE_DIM, 64, 3)
    policy._optimizer = torch.optim.SGD(policy._model.parameters(), lr=0.01)
    return policy


class TestGossipNode:
    def test_send_update_norm_clipped(self):
        bus: dict[str, list] = {}
        policy_a = _get_fresh_policy()
        transport_a = InMemoryTransport("A", bus)
        node_a = GossipNode(
            node_id="A",
            policy=policy_a,
            transport=transport_a,
            max_norm=0.5,
        )

        # Register B as peer of A (A sends to B via A's own transport)
        node_a.register_peer("B", transport_a)

        # Modify A's policy parameters to create a large delta
        params = policy_a.get_parameters()
        policy_a.set_parameters(params + 10.0)

        node_a.send_update()

        # Check what was sent to B's inbox
        messages = bus.get("B", [])
        assert len(messages) >= 1
        _, delta = messages[0]
        norm = float(np.linalg.norm(delta))
        assert norm <= 0.5 + 1e-6

    def test_aggregate_applies_update(self):
        bus: dict[str, list] = {}
        policy = _get_fresh_policy()
        transport = InMemoryTransport("A", bus)
        node = GossipNode(node_id="A", policy=policy, transport=transport)
        params_before = policy.get_parameters().copy()

        delta = np.ones_like(params_before) * 0.01
        node.receive_update("B", delta)
        node.aggregate()

        params_after = policy.get_parameters()
        assert not np.allclose(params_before, params_after)

    def test_multi_node_gossip_round(self):
        """3 nodes do a gossip round: send updates, aggregate."""
        bus: dict[str, list] = {}

        nodes = {}
        for name in ["A", "B", "C"]:
            policy = _get_fresh_policy()
            transport = InMemoryTransport(name, bus)
            nodes[name] = GossipNode(
                node_id=name,
                policy=policy,
                transport=transport,
                max_norm=1.0,
            )

        # Register peers (fully connected)
        for name, node in nodes.items():
            for peer_name in nodes:
                if peer_name != name:
                    node.register_peer(peer_name, node._transport)

        # Train each node differently
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        for i, (name, node) in enumerate(nodes.items()):
            for _ in range(5):
                node._policy.update(features, action_id=i % 3, reward=float(i))

        params_before = {name: node._policy.get_parameters().copy() for name, node in nodes.items()}

        # Gossip round: all send, then all aggregate
        for node in nodes.values():
            node.send_update()
        for node in nodes.values():
            node.aggregate()

        changed = any(
            not np.allclose(params_before[name], node._policy.get_parameters())
            for name, node in nodes.items()
        )
        assert changed

    def test_step_triggers_at_interval(self):
        bus: dict[str, list] = {}
        policy = _get_fresh_policy()
        transport = InMemoryTransport("A", bus)
        node_a = GossipNode(
            node_id="A",
            policy=policy,
            transport=transport,
            gossip_interval=5,
        )
        node_a.register_peer("B", transport)

        # Modify policy to have non-zero delta
        params = policy.get_parameters()
        policy.set_parameters(params + 0.1)

        # Steps 1-4: no gossip
        for _ in range(4):
            node_a.step()
        assert len(bus.get("B", [])) == 0

        # Step 5: gossip triggered
        node_a.step()
        assert len(bus.get("B", [])) > 0

    def test_krum_filters_adversarial_in_gossip(self):
        """Adversarial node sends large delta; Krum should protect the honest node."""
        bus: dict[str, list] = {}
        policy = _get_fresh_policy()
        transport = InMemoryTransport("honest", bus)
        honest_node = GossipNode(
            node_id="honest",
            policy=policy,
            transport=transport,
            max_norm=1.0,
        )

        params_before = policy.get_parameters().copy()

        # 3 honest updates (small deltas near zero)
        rng = np.random.default_rng(42)
        param_size = len(params_before)
        for i in range(3):
            small_delta = rng.normal(0, 0.01, param_size).astype(np.float32)
            honest_node.receive_update(f"honest_peer_{i}", small_delta)

        # 1 adversarial update (huge delta)
        adversarial_delta = np.ones(param_size, dtype=np.float32) * 100.0
        honest_node.receive_update("adversary", adversarial_delta)

        honest_node.aggregate()

        params_after = policy.get_parameters()
        total_change = float(np.linalg.norm(params_after - params_before))
        assert total_change < 1.0


# ─── Differential Privacy tests ──────────────────────────────────────


class TestDPNoise:
    def test_dp_noise_adds_noise(self):
        """_add_dp_noise() should change the delta vector."""
        delta = np.ones(10, dtype=np.float64) * 0.5
        noisy = _add_dp_noise(delta, max_norm=1.0, epsilon=1.0, delta_dp=1e-5)
        assert not np.allclose(noisy, delta), "DP noise should change the delta"

    def test_dp_noise_scale_proportional_to_max_norm(self):
        """Higher max_norm → larger noise variance."""
        np.random.default_rng(0)
        delta = np.zeros(1000, dtype=np.float64)

        # Run many samples and compare empirical std
        n_samples = 50
        std_low = np.std(
            [
                _add_dp_noise(delta.copy(), max_norm=0.1, epsilon=1.0, delta_dp=1e-5)
                for _ in range(n_samples)
            ]
        )
        std_high = np.std(
            [
                _add_dp_noise(delta.copy(), max_norm=10.0, epsilon=1.0, delta_dp=1e-5)
                for _ in range(n_samples)
            ]
        )
        assert std_high > std_low, "Larger max_norm should produce larger noise"

    def test_dp_disabled_no_noise(self):
        """With dp_enabled=False, send_update sends clipped delta without DP noise."""
        bus: dict[str, list] = {}
        policy_a = _get_fresh_policy()
        transport_a = InMemoryTransport("A", bus)
        node_a = GossipNode(
            node_id="A",
            policy=policy_a,
            transport=transport_a,
            max_norm=0.5,
            dp_enabled=False,
        )
        node_a.register_peer("B", transport_a)

        # Set parameters to a known delta
        params = policy_a.get_parameters()
        policy_a.set_parameters(params + 10.0)
        node_a.send_update()

        messages = bus.get("B", [])
        assert len(messages) >= 1
        _, delta = messages[0]
        # With DP disabled, delta should be exactly clipped (norm <= 0.5)
        norm = float(np.linalg.norm(delta))
        assert norm <= 0.5 + 1e-6, f"Expected clipped norm <= 0.5, got {norm}"

    def test_dp_enabled_noisy_delta(self):
        """With dp_enabled=True, the delta received by a peer differs from exact clipped delta."""
        bus: dict[str, list] = {}
        policy_a = _get_fresh_policy()
        transport_a = InMemoryTransport("A", bus)
        node_a = GossipNode(
            node_id="A",
            policy=policy_a,
            transport=transport_a,
            max_norm=1.0,
            dp_enabled=True,
            dp_epsilon=0.5,
            dp_delta=1e-5,
        )
        node_a.register_peer("B", transport_a)

        params = policy_a.get_parameters()
        policy_a.set_parameters(params + 5.0)

        # Compute what the exact clipped delta would be
        current = policy_a.get_parameters()
        raw_delta = current - node_a._base_params
        exact_clipped = _clip_norm(raw_delta, 1.0)

        node_a.send_update()

        messages = bus.get("B", [])
        assert len(messages) >= 1
        _, received_delta = messages[0]
        # With DP enabled, the received delta should differ from exact clipped
        assert not np.allclose(received_delta, exact_clipped, atol=1e-6), (
            "DP-enabled send_update should add noise to the clipped delta"
        )

    def test_rejected_peer_deltas_dropped(self):
        """Manually rejected peers should have their deltas dropped during aggregate."""
        bus: dict[str, list] = {}
        policy = _get_fresh_policy()
        transport = InMemoryTransport("A", bus)
        node = GossipNode(node_id="A", policy=policy, transport=transport)

        params_before = policy.get_parameters().copy()
        large_delta = np.ones_like(params_before) * 5.0

        # Manually mark the peer as rejected
        node._rejected_peers.add("bad_peer")

        # Put a message in the transport inbox from bad_peer
        bus["A"].append(("bad_peer", large_delta))

        node.aggregate()

        params_after = policy.get_parameters()
        # The large delta from bad_peer should be dropped — params should not change much
        assert np.allclose(params_before, params_after, atol=1e-6), (
            "Rejected peer's delta should not affect policy parameters"
        )


# ─── Node Rejection (Rule Hash) tests ────────────────────────────────


class TestNodeRejection:
    def _make_honest_node(
        self,
        node_id: str,
        bus: dict,
        rule_hash_bus: dict,
        config: SecurityConfig,
    ) -> GossipNode:
        policy = _get_fresh_policy()
        transport = InMemoryTransport(node_id, bus, rule_hash_bus)
        verifier = RuleVerifier(config)
        return GossipNode(
            node_id=node_id,
            policy=policy,
            transport=transport,
            max_norm=1.0,
            dp_enabled=False,
            rule_verifier=verifier,
        )

    def test_tampered_node_rejected(self):
        """Honest nodes should reject deltas from a tampered node after rule hash exchange."""
        bus: dict = {}
        rule_hash_bus: dict = {}
        honest_config = SecurityConfig()
        tampered_config = SecurityConfig(no_harm_to_humans=False)

        # 3 honest nodes + 1 tampered node
        nodes = {
            name: self._make_honest_node(name, bus, rule_hash_bus, honest_config)
            for name in ["A", "B", "C"]
        }
        tampered_policy = _get_fresh_policy()
        tampered_transport = InMemoryTransport("T", bus, rule_hash_bus)
        tampered_verifier = RuleVerifier(tampered_config)
        tampered_node = GossipNode(
            node_id="T",
            policy=tampered_policy,
            transport=tampered_transport,
            max_norm=1.0,
            dp_enabled=False,
            rule_verifier=tampered_verifier,
        )

        all_nodes = {**nodes, "T": tampered_node}

        # Fully connect all nodes
        for name, node in all_nodes.items():
            for peer_name, peer_node in all_nodes.items():
                if peer_name != name:
                    node.register_peer(peer_name, node._transport)

        # Exchange rule hashes: each node sends its hash to all peers
        for node in all_nodes.values():
            node.send_rule_hash()

        # Each honest node aggregates: should reject tampered node
        for node in nodes.values():
            # Put a small delta from tampered node in the bus
            bus[node.node_id].append(("T", np.ones(len(tampered_policy.get_parameters())) * 100.0))
            node.aggregate()
            assert "T" in node._rejected_peers, (
                f"Node {node.node_id} should have rejected tampered node T"
            )

    def test_honest_nodes_still_aggregate(self):
        """After rejecting tampered node, honest nodes still aggregate among themselves."""
        bus: dict = {}
        rule_hash_bus: dict = {}
        honest_config = SecurityConfig()

        nodes = {
            name: self._make_honest_node(name, bus, rule_hash_bus, honest_config)
            for name in ["A", "B", "C"]
        }

        # Connect all to each other
        for name, node in nodes.items():
            for peer_name, peer_node in nodes.items():
                if peer_name != name:
                    node.register_peer(peer_name, node._transport)

        # Modify B and C parameters so A receives non-trivial deltas
        params = nodes["B"]._policy.get_parameters()
        nodes["B"]._policy.set_parameters(params + 0.1)
        nodes["C"]._policy.set_parameters(params + 0.2)

        # Exchange rule hashes and updates
        for node in nodes.values():
            node.send_rule_hash()
            node.send_update()

        params_before = nodes["A"]._policy.get_parameters().copy()
        nodes["A"].aggregate()
        params_after = nodes["A"]._policy.get_parameters()

        # A should have changed params (received updates from B and C)
        assert not np.allclose(params_before, params_after, atol=1e-8), (
            "Honest node A should aggregate updates from honest peers B and C"
        )

    def test_rule_hash_exchange_via_in_memory_transport(self):
        """InMemoryTransport correctly passes rule hashes between nodes."""
        bus: dict = {}
        rule_hash_bus: dict = {}
        config = SecurityConfig()

        t_a = InMemoryTransport("A", bus, rule_hash_bus)
        t_b = InMemoryTransport("B", bus, rule_hash_bus)

        verifier = RuleVerifier(config)
        t_a.send_rule_hash("B", verifier.rule_hash_bytes)

        received = t_b.receive_rule_hashes()
        assert len(received) == 1
        sender_id, hash_bytes = received[0]
        assert sender_id == "A"
        assert verifier.verify(hash_bytes), "Received hash should verify correctly"

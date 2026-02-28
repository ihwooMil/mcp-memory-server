"""Immutable rule hash verification for P2P node integrity.

Provides:
- RuleVerifier: computes and verifies SHA-256 hashes of immutable rules
"""

from __future__ import annotations

import hashlib
import logging

from aimemory.config import SecurityConfig

logger = logging.getLogger(__name__)


class RuleVerifier:
    """Computes and verifies SHA-256 hashes of immutable security rules.

    Each gossip node carries a RuleVerifier. Before accepting parameter
    deltas from a peer, the node checks that the peer's rule hash matches
    its own. Mismatches indicate rule tampering.
    """

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config
        self._hash = self._compute_hash(config)

    @staticmethod
    def _compute_hash(config: SecurityConfig) -> str:
        """Compute deterministic SHA-256 hash of SecurityConfig fields."""
        fields = sorted(config.model_dump().items())
        serialized = "|".join(f"{k}={v}" for k, v in fields)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def rule_hash(self) -> str:
        """Return the hex digest of the current rule hash."""
        return self._hash

    @property
    def rule_hash_bytes(self) -> bytes:
        """Return the rule hash as raw bytes (for transport)."""
        return self._hash.encode("utf-8")

    def verify(self, peer_hash: str | bytes) -> bool:
        """Verify a peer's rule hash against our own.

        Args:
            peer_hash: The peer's rule hash (str hex digest or bytes).

        Returns:
            True if hashes match, False if tampered.
        """
        if isinstance(peer_hash, bytes):
            peer_hash = peer_hash.decode("utf-8")
        return peer_hash == self._hash

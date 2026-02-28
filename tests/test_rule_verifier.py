"""Tests for RuleVerifier: immutable rule hash verification."""

from __future__ import annotations

from aimemory.config import SecurityConfig
from aimemory.online.rule_verifier import RuleVerifier


class TestRuleVerifier:
    def test_same_config_same_hash(self):
        """Two RuleVerifier instances with identical SecurityConfig produce the same hash."""
        config_a = SecurityConfig()
        config_b = SecurityConfig()
        verifier_a = RuleVerifier(config_a)
        verifier_b = RuleVerifier(config_b)
        assert verifier_a.rule_hash == verifier_b.rule_hash

    def test_different_config_different_hash(self):
        """Altered SecurityConfig produces a different hash."""
        honest_config = SecurityConfig()
        tampered_config = SecurityConfig(no_harm_to_humans=False)
        verifier_honest = RuleVerifier(honest_config)
        verifier_tampered = RuleVerifier(tampered_config)
        assert verifier_honest.rule_hash != verifier_tampered.rule_hash

    def test_verify_matching(self):
        """verify() returns True for matching peer hash."""
        config = SecurityConfig()
        verifier_a = RuleVerifier(config)
        verifier_b = RuleVerifier(config)
        assert verifier_a.verify(verifier_b.rule_hash)

    def test_verify_mismatch(self):
        """verify() returns False for mismatched peer hash."""
        honest_config = SecurityConfig()
        tampered_config = SecurityConfig(respect_life_dignity=False)
        verifier_honest = RuleVerifier(honest_config)
        verifier_tampered = RuleVerifier(tampered_config)
        assert not verifier_honest.verify(verifier_tampered.rule_hash)

    def test_hash_deterministic(self):
        """Same config always produces same hash (called twice)."""
        config = SecurityConfig()
        verifier = RuleVerifier(config)
        hash1 = verifier.rule_hash
        hash2 = verifier.rule_hash
        assert hash1 == hash2

    def test_verify_bytes_input(self):
        """verify() accepts bytes (from transport) as well as str."""
        config = SecurityConfig()
        verifier = RuleVerifier(config)
        assert verifier.verify(verifier.rule_hash_bytes)

    def test_verify_bytes_mismatch(self):
        """verify() returns False for mismatched bytes hash."""
        honest_config = SecurityConfig()
        tampered_config = SecurityConfig(no_harm_to_humans=False)
        verifier_honest = RuleVerifier(honest_config)
        verifier_tampered = RuleVerifier(tampered_config)
        assert not verifier_honest.verify(verifier_tampered.rule_hash_bytes)

    def test_hash_is_sha256_hex(self):
        """The rule hash should be a 64-char hex string (SHA-256)."""
        config = SecurityConfig()
        verifier = RuleVerifier(config)
        assert len(verifier.rule_hash) == 64
        int(verifier.rule_hash, 16)  # should not raise

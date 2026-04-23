# tests/test_models.py
# Proves that our Pydantic models reject invalid data at the boundary.
#
# Why this matters: without field validation, a malformed API response
# (confidence=1.5, decision="maybe") passes silently through the pipeline
# and corrupts logs and memory. These tests confirm the models catch that.

import pytest
from pydantic import ValidationError
from models import GuardrailResult, FailureRecord, DecisionLog


# ── GuardrailResult ────────────────────────────────────────────────────────────

class TestGuardrailResult:
    def test_valid_allow(self):
        r = GuardrailResult(
            decision="allow",
            reason="Safe action.",
            confidence=0.95,
            action_type="data_display",
        )
        assert r.decision == "allow"
        assert r.confidence == 0.95

    def test_valid_block(self):
        r = GuardrailResult(
            decision="block",
            reason="Risky action.",
            confidence=0.99,
            action_type="credential_access",
        )
        assert r.decision == "block"

    def test_invalid_decision_rejected(self):
        """decision must be exactly 'allow' or 'block'."""
        with pytest.raises(ValidationError):
            GuardrailResult(
                decision="maybe",    # invalid
                reason="Unsure.",
                confidence=0.5,
                action_type="other",
            )

    def test_confidence_above_one_rejected(self):
        """confidence > 1.0 is out of range and must be rejected."""
        with pytest.raises(ValidationError):
            GuardrailResult(
                decision="block",
                reason="Risky.",
                confidence=1.5,      # invalid
                action_type="other",
            )

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            GuardrailResult(
                decision="allow",
                reason="Safe.",
                confidence=-0.1,     # invalid
                action_type="other",
            )

    def test_boundary_confidence_zero(self):
        """0.0 is the minimum valid confidence."""
        r = GuardrailResult(decision="allow", reason="Safe.", confidence=0.0, action_type="other")
        assert r.confidence == 0.0

    def test_boundary_confidence_one(self):
        """1.0 is the maximum valid confidence."""
        r = GuardrailResult(decision="block", reason="Risky.", confidence=1.0, action_type="other")
        assert r.confidence == 1.0


# ── FailureRecord ──────────────────────────────────────────────────────────────

class TestFailureRecord:
    def test_default_distance(self):
        """distance defaults to 0.0 when a record is first stored."""
        r = FailureRecord(
            action="Delete all files.",
            risk_reason="Destructive operation.",
            action_type="file_deletion",
            run_id="test123",
        )
        assert r.distance == 0.0

    def test_distance_populated_on_retrieval(self):
        """distance is set to the actual cosine distance when retrieved."""
        r = FailureRecord(
            action="Delete all files.",
            risk_reason="Destructive operation.",
            action_type="file_deletion",
            run_id="test123",
            distance=0.1234,
        )
        assert r.distance == 0.1234


# ── DecisionLog ───────────────────────────────────────────────────────────────

class TestDecisionLog:
    def _valid_log(self, **overrides):
        defaults = dict(
            timestamp="2026-01-01T00:00:00+00:00",
            action="Do something.",
            decision="allow",
            reason="Safe.",
            confidence=0.9,
            action_type="other",
            source="guardrail",
            run_id="abc12345",
        )
        defaults.update(overrides)
        return DecisionLog(**defaults)

    def test_valid_guardrail_source(self):
        log = self._valid_log(source="guardrail")
        assert log.source == "guardrail"
        assert log.similarity_distance == -1.0   # default

    def test_valid_memory_source(self):
        log = self._valid_log(source="memory", similarity_distance=0.12)
        assert log.source == "memory"
        assert log.similarity_distance == 0.12

    def test_valid_fallback_source(self):
        log = self._valid_log(source="fallback", decision="block")
        assert log.source == "fallback"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_log(source="unknown")   # invalid

    def test_invalid_decision_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_log(decision="pending")  # invalid

    def test_latency_defaults_to_zero(self):
        log = self._valid_log()
        assert log.latency_ms == 0.0

    def test_category_optional(self):
        log = self._valid_log()
        assert log.category is None

        log2 = self._valid_log(category="safe_baseline")
        assert log2.category == "safe_baseline"

    def test_roundtrip_json(self):
        """Serialise to JSON and back — Pydantic v2 API."""
        log = self._valid_log(source="memory", similarity_distance=0.08, latency_ms=12.5)
        json_str = log.model_dump_json()
        restored = DecisionLog.model_validate_json(json_str)
        assert restored.similarity_distance == 0.08
        assert restored.latency_ms == 12.5

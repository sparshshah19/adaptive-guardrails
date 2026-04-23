# tests/test_detector.py
# Tests the confidence threshold boundary in detector.py.
#
# The threshold (BLOCK_CONFIDENCE_THRESHOLD = 0.6) controls which blocked
# actions get stored in memory. Testing the exact boundary matters because:
#   - Too permissive: low-confidence blocks pollute memory with false positives
#   - Too strict: real threats get missed
# These tests confirm that the boundary is exactly where config.py says it is.

from config import BLOCK_CONFIDENCE_THRESHOLD
from detector import should_store
from models import GuardrailResult


def _result(decision: str, confidence: float, action_type: str = "other") -> GuardrailResult:
    return GuardrailResult(
        decision=decision,
        reason="Test reason.",
        confidence=confidence,
        action_type=action_type,
    )


class TestShouldStore:
    def test_high_confidence_block_stored(self):
        """A block above threshold should be stored."""
        assert should_store(_result("block", 0.99)) is True

    def test_block_just_above_threshold_stored(self):
        """Just above the boundary — should store."""
        assert should_store(_result("block", BLOCK_CONFIDENCE_THRESHOLD + 0.01)) is True

    def test_block_at_threshold_not_stored(self):
        """At exactly the threshold — should NOT store (strictly greater than)."""
        assert should_store(_result("block", BLOCK_CONFIDENCE_THRESHOLD)) is False

    def test_block_below_threshold_not_stored(self):
        """Below threshold — uncertain block should not pollute memory."""
        assert should_store(_result("block", 0.4)) is False

    def test_allow_never_stored(self):
        """An allow decision is never stored, regardless of confidence."""
        assert should_store(_result("allow", 0.99)) is False
        assert should_store(_result("allow", 0.0)) is False

    def test_low_confidence_allow_not_stored(self):
        assert should_store(_result("allow", 0.1)) is False

    def test_action_type_does_not_affect_storage(self):
        """All action types are treated equally by should_store."""
        for at in ["file_download", "credential_access", "script_execution", "other"]:
            assert should_store(_result("block", 0.95, action_type=at)) is True

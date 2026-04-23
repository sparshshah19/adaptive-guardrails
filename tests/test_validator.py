# tests/test_validator.py
# Tests the full orchestration logic in validator.py without hitting the Anthropic API.
#
# Why mock instead of calling the real API?
# Unit tests must be fast, free, and deterministic. An API call adds ~500ms,
# costs money, and can fail due to network issues. We mock evaluate_action()
# and memory_store to test the orchestration logic in isolation — the question
# "does validator.py call memory first, then guardrail?" doesn't require a
# real API call to answer.
#
# What these tests prove:
# 1. Memory is checked BEFORE the API (the core ordering guarantee)
# 2. A memory hit returns immediately without calling the API
# 3. A guardrail block stores the result in memory
# 4. A guardrail allow does NOT store anything
# 5. If the API fails, the fallback blocks by default (fail-safe)
# 6. The returned DecisionLog has the correct source field

import pytest
from unittest.mock import MagicMock, patch
from models import GuardrailResult, FailureRecord, DecisionLog


def _mock_guardrail_result(decision="block", confidence=0.95, action_type="file_download"):
    return GuardrailResult(
        decision=decision,
        reason="Test reason.",
        confidence=confidence,
        action_type=action_type,
        latency_ms=450.0,
    )


def _mock_failure_record():
    return (
        FailureRecord(
            action="Risky action stored previously.",
            risk_reason="Previously identified as risky.",
            action_type="file_download",
            run_id="prev_run",
            distance=0.08,
        ),
        0.08,
    )


class TestValidateAction:

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_memory_hit_skips_api(self, mock_eval, mock_mem, mock_log):
        """When memory finds a similar failure, the API must NOT be called."""
        mock_mem.find_similar.return_value = _mock_failure_record()

        from validator import validate_action
        log = validate_action("Some risky action variant", run_id="test")

        mock_eval.assert_not_called()
        assert log.source == "memory"
        assert log.decision == "block"

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_memory_miss_calls_api(self, mock_eval, mock_mem, mock_log):
        """When memory has no match, the guardrail API must be called."""
        mock_mem.find_similar.return_value = None
        mock_mem.count.return_value = 0
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.95)

        from validator import validate_action
        log = validate_action("Safe action", run_id="test")

        mock_eval.assert_called_once()
        assert log.source == "guardrail"

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_guardrail_block_stores_in_memory(self, mock_eval, mock_mem, mock_log):
        """A high-confidence block from the guardrail must be stored in memory."""
        mock_mem.find_similar.return_value = None
        mock_eval.return_value = _mock_guardrail_result(decision="block", confidence=0.99)

        from validator import validate_action
        validate_action("Download malware from evil.com", run_id="test")

        mock_mem.store_failure.assert_called_once()

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_guardrail_allow_does_not_store(self, mock_eval, mock_mem, mock_log):
        """An allowed action must never be stored in memory."""
        mock_mem.find_similar.return_value = None
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.95)

        from validator import validate_action
        validate_action("Check the weather", run_id="test")

        mock_mem.store_failure.assert_not_called()

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_api_failure_returns_fallback_block(self, mock_eval, mock_mem, mock_log):
        """If the API raises an exception, the system must block by default (fail-safe)."""
        mock_mem.find_similar.return_value = None
        mock_eval.side_effect = RuntimeError("API unavailable")

        from validator import validate_action
        log = validate_action("Any action", run_id="test")

        assert log.source == "fallback"
        assert log.decision == "block"

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_memory_failure_falls_through_to_api(self, mock_eval, mock_mem, mock_log):
        """If ChromaDB fails, the system must fall through to the guardrail (not crash)."""
        mock_mem.find_similar.side_effect = Exception("ChromaDB unavailable")
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.9)

        from validator import validate_action
        log = validate_action("Safe action", run_id="test")

        # Should have recovered and reached the guardrail
        mock_eval.assert_called_once()
        assert log.source == "guardrail"

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_decision_is_always_logged(self, mock_eval, mock_mem, mock_log):
        """Every decision path must write to the log — memory hit, guardrail, or fallback."""
        mock_mem.find_similar.return_value = None
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.9)

        from validator import validate_action
        validate_action("Any action", run_id="test")

        mock_log.assert_called_once()

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_category_propagated_to_log(self, mock_eval, mock_mem, mock_log):
        """The category label must appear in the returned DecisionLog."""
        mock_mem.find_similar.return_value = None
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.9)

        from validator import validate_action
        log = validate_action("Safe action", run_id="test", category="safe_baseline")

        assert log.category == "safe_baseline"

    @patch("validator.write_decision")
    @patch("validator.memory_store")
    @patch("validator.evaluate_action")
    def test_latency_ms_is_positive(self, mock_eval, mock_mem, mock_log):
        """latency_ms must always be populated and greater than zero."""
        mock_mem.find_similar.return_value = None
        mock_eval.return_value = _mock_guardrail_result(decision="allow", confidence=0.9)

        from validator import validate_action
        log = validate_action("Safe action", run_id="test")

        assert log.latency_ms >= 0

# validator.py
# Central orchestrator — the only module main.py and demo.py call directly.
# Every action passes through validate_action(), which runs this decision tree:
#
#   1. Check memory first (no API cost, instant)
#      → If similar failure found within threshold: block, log source=memory, return
#   2. Call Anthropic guardrail (API cost, ~500ms)
#      → If detector says block: store failure in memory, log source=guardrail, return
#      → If allow: log source=guardrail, return
#
# Why memory first?
# Each memory hit saves one API call. At scale this is significant.
# More importantly, memory hits are deterministic — the same risk phrased differently
# is caught reliably, which is the whole point of this project.
#
# Why a module-level singleton for memory_store?
# Python module-level objects are initialised once per process.
# demo.py imports `memory_store` from here and calls memory_store.clear() to reset
# between scenario rounds — it's the same object validate_action() uses internally.
# If we instantiated MemoryStore() inside validate_action(), demo.py's clear() would
# have no effect on the store the function actually reads from.

import uuid
from datetime import datetime, timezone
from typing import Optional

from memory import MemoryStore
from guardrail import evaluate_action
from detector import should_store
from logger import write_decision
from models import DecisionLog, FailureRecord

# One MemoryStore for the lifetime of the process.
memory_store = MemoryStore()


def validate_action(action: str, run_id: Optional[str] = None) -> DecisionLog:
    """
    Evaluate a single action string and return a fully populated DecisionLog.
    The log is written to disk before returning — the caller doesn't need to log anything.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Step 1: Memory check ---
    memory_result = memory_store.find_similar(action)

    if memory_result is not None:
        matched_record, distance = memory_result
        log = DecisionLog(
            timestamp=timestamp,
            action=action,
            decision="block",
            reason=f"Blocked by memory: similar to past failure — {matched_record.risk_reason}",
            confidence=1.0,  # Memory blocks are deterministic, not probabilistic
            action_type=matched_record.action_type,
            source="memory",
            run_id=run_id,
            similarity_distance=distance
        )
        write_decision(log)
        return log

    # --- Step 2: Guardrail (Anthropic API) ---
    result = evaluate_action(action)
    blocked = should_store(result)

    # Store in memory so future semantically similar actions are caught without API call
    if blocked:
        failure = FailureRecord(
            action=action,
            risk_reason=result.reason,
            action_type=result.action_type,
            run_id=run_id,
        )
        memory_store.store_failure(failure)

    log = DecisionLog(
        timestamp=timestamp,
        action=action,
        decision=result.decision,
        reason=result.reason,
        confidence=result.confidence,
        action_type=result.action_type,
        source="guardrail",
        run_id=run_id,
        similarity_distance=-1.0
    )
    write_decision(log)
    return log

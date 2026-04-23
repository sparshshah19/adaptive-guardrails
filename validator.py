# validator.py
# Central orchestrator — the only module main.py, demo.py, and api.py call directly.
#
# Decision pipeline for every action:
#   1. Check memory    (~10ms, no API cost)
#      → Hit within threshold → block, log source=memory, return
#   2. Neural Net      (~2ms, no API cost)
#      → prob > 0.98 → block, log source=classifier, return
#      → prob < 0.02 → allow, log source=classifier, return
#      → 0.02–0.98   → uncertain, fall through to Claude
#      → Not ready (no model file yet) → fall through to Claude silently
#   3. Anthropic guardrail (~400–600ms, API cost)
#      → High-confidence block → store in memory, log source=guardrail, return
#      → Allow or low-confidence block → log source=guardrail, return
#   4. Fallback (if both memory AND API fail)
#      → Block by default — failing open is worse than failing closed for a safety system
#      → log source=fallback, return
#
# Why a module-level singleton for memory_store and risk_classifier?
# Python module objects are initialised once per process. demo.py and api.py import
# `memory_store` from here and call memory_store.clear() directly — it's the same
# object validate_action() reads from internally. Per-call instantiation would mean
# demo.py's clear() has no effect on what the function actually sees.
# Same logic applies to risk_classifier — load() is called once at startup.

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from memory import MemoryStore
from guardrail import evaluate_action
from detector import should_store
from logger import write_decision
from models import DecisionLog, FailureRecord
from classifier import risk_classifier, ClassifierNotReady
from config import CLASSIFIER_MODEL_PATH, EMBEDDING_MODEL

# Shared across the process lifetime.
memory_store = MemoryStore()

# Inject the shared encoder (same model instance memory.py uses) then load weights.
# is_ready() requires encoder to be set — without this it always returns False.
if risk_classifier._encoder is None:
    from sentence_transformers import SentenceTransformer
    risk_classifier._encoder = SentenceTransformer(EMBEDDING_MODEL)
risk_classifier.load(CLASSIFIER_MODEL_PATH)


def validate_action(
    action: str,
    run_id: Optional[str] = None,
    category: Optional[str] = None
) -> DecisionLog:
    """
    Evaluate a single action string. Returns a fully populated DecisionLog.
    Writes the log to disk before returning — callers don't need to log anything.

    Parameters:
        action:   The action string to evaluate.
        run_id:   Optional identifier to group related actions (e.g. per demo round).
        category: Optional label from agent.py (e.g. "safe_baseline") for eval breakdowns.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Memory check ──────────────────────────────────────────────────
    try:
        memory_result = memory_store.find_similar(action)
    except Exception:
        # ChromaDB failure — fall through to next stage rather than crashing.
        memory_result = None

    if memory_result is not None:
        matched_record, distance = memory_result
        latency_ms = (time.perf_counter() - start) * 1000
        log = DecisionLog(
            timestamp=timestamp,
            action=action,
            decision="block",
            reason=f"Blocked by memory: similar to past failure — {matched_record.risk_reason}",
            confidence=1.0,       # memory blocks are deterministic, not probabilistic
            action_type=matched_record.action_type,
            source="memory",
            run_id=run_id,
            similarity_distance=distance,
            latency_ms=latency_ms,
            category=category,
        )
        write_decision(log)
        return log

    # ── Step 2: Neural Net classifier ────────────────────────────────────────
    # The NN handles ~80% of decisions after training, at ~2ms with no API cost.
    # We only skip it if the model file hasn't been created yet (first-run cold start).
    classifier_used = False
    try:
        nn_decision, nn_prob = risk_classifier.predict(action)
        classifier_used = True

        if nn_decision in ("block", "allow"):
            # High-confidence NN decision — return immediately, no API call needed.
            latency_ms = (time.perf_counter() - start) * 1000
            log = DecisionLog(
                timestamp=timestamp,
                action=action,
                decision=nn_decision,
                reason=(
                    f"Neural classifier: {'risk' if nn_decision == 'block' else 'safe'} "
                    f"(probability={nn_prob:.3f})"
                ),
                confidence=nn_prob if nn_decision == "block" else (1.0 - nn_prob),
                action_type="classifier",
                source="classifier",
                run_id=run_id,
                similarity_distance=-1.0,
                latency_ms=latency_ms,
                category=category,
                classifier_used=True,
            )
            write_decision(log)
            return log

        # nn_decision == "uncertain" — fall through to Claude

    except ClassifierNotReady:
        # Model not trained yet. This is expected on first run.
        # Falls through to Claude silently — no error logged.
        pass
    except Exception:
        # Any other classifier error (corrupt weights, etc.) — fall through to Claude.
        pass

    # ── Step 3: Anthropic guardrail ───────────────────────────────────────────
    try:
        result = evaluate_action(action)
        blocked = should_store(result)

        if blocked:
            try:
                failure = FailureRecord(
                    action=action,
                    risk_reason=result.reason,
                    action_type=result.action_type,
                    run_id=run_id,
                )
                memory_store.store_failure(failure)
            except Exception:
                pass   # storage failure should not block the decision from being returned

        latency_ms = (time.perf_counter() - start) * 1000
        log = DecisionLog(
            timestamp=timestamp,
            action=action,
            decision=result.decision,
            reason=result.reason,
            confidence=result.confidence,
            action_type=result.action_type,
            source="guardrail",
            run_id=run_id,
            similarity_distance=-1.0,
            latency_ms=latency_ms,
            category=category,
            classifier_used=classifier_used,
        )
        write_decision(log)
        return log

    except Exception as e:
        # ── Step 4: Fallback ──────────────────────────────────────────────────
        # Both memory and API failed. Block by default — a safety system should
        # never fail open. Log the failure source so eval.py can flag it.
        latency_ms = (time.perf_counter() - start) * 1000
        log = DecisionLog(
            timestamp=timestamp,
            action=action,
            decision="block",
            reason=f"Blocked by fallback: guardrail unavailable ({type(e).__name__})",
            confidence=1.0,
            action_type="unknown",
            source="fallback",
            run_id=run_id,
            similarity_distance=-1.0,
            latency_ms=latency_ms,
            category=category,
            classifier_used=classifier_used,
        )
        write_decision(log)
        return log

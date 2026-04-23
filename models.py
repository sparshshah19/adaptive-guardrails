# models.py
# Shared Pydantic v2 data models used across the entire system.
# Centralised here to prevent circular imports between guardrail, memory, logger, and validator.
#
# Pydantic v2 enforces these constraints at runtime — if guardrail.py returns
# confidence=1.5 or decision="maybe", Pydantic raises a ValidationError immediately
# instead of silently passing bad data through the pipeline.

from typing import Literal, Optional
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """
    The structured output from one Anthropic API call.

    decision:    Enforced as exactly "allow" or "block" — no other values accepted.
    confidence:  Must be between 0.0 and 1.0. Pydantic rejects out-of-range values.
    action_type: Category label extracted by the model (e.g. "file_download").
                 Used for per-category breakdowns in eval.py.
    latency_ms:  How long the API call took. Populated by guardrail.py.
    """
    decision: Literal["allow", "block"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    action_type: str
    latency_ms: float = 0.0


class FailureRecord(BaseModel):
    """
    One row stored in ChromaDB's 'blocked_actions' collection.
    Written by memory.store_failure() whenever detector.should_store() returns True.

    distance: 0.0 when first stored (the record itself has no distance — it IS the reference).
              Populated with the actual cosine distance when retrieved by find_similar().
    """
    action: str
    risk_reason: str
    action_type: str
    run_id: str
    distance: float = 0.0


class DecisionLog(BaseModel):
    """
    One line in the JSONL log. Written for every action, regardless of outcome.

    source:
      "guardrail"   — Anthropic API was called. latency_ms reflects API round-trip.
      "memory"      — ChromaDB returned a similar past failure. No API call made.
      "classifier"  — PyTorch NN made a high-confidence decision. No API call made.
      "fallback"    — Both memory and API failed. Action was blocked as a safe default.

    similarity_distance:
      -1.0 when source is "guardrail" or "fallback" (not applicable).
      Actual cosine distance (0.0–2.0) when source is "memory".

    latency_ms:
      Total time for the decision in milliseconds.
      Memory hits are ~10ms. Guardrail calls are ~400–600ms.
      This is what makes the latency savings claim measurable.
    """
    timestamp: str
    action: str
    decision: Literal["allow", "block"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    action_type: str
    source: Literal["guardrail", "memory", "classifier", "fallback"]
    run_id: str
    similarity_distance: float = -1.0
    latency_ms: float = 0.0
    category: Optional[str] = None   # populated by main.py for per-category eval breakdown
    classifier_used: bool = False     # True when the NN made or contributed to the decision

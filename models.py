# models.py
# Shared Pydantic v2 data models used across the entire system.
# Centralised here to prevent circular imports between guardrail, memory, logger, and validator.

from pydantic import BaseModel


class GuardrailResult(BaseModel):
    """
    The structured output returned by guardrail.py after one Anthropic API call.
    decision: "allow" or "block"
    confidence: 0.0–1.0. Only blocks above 0.6 are stored in memory (see detector.py).
    action_type: a category label the model extracts, e.g. "file_download", "form_submission".
                 This ends up in ChromaDB metadata and makes eval.py breakdowns meaningful.
    """
    decision: str
    reason: str
    confidence: float
    action_type: str


class FailureRecord(BaseModel):
    """
    One row in the ChromaDB 'blocked_actions' collection.
    Stored whenever detector.py returns True.
    distance is 0.0 when the record is first written;
    it gets populated with the actual cosine distance when retrieved by memory.find_similar().
    """
    action: str
    risk_reason: str
    action_type: str
    run_id: str
    distance: float = 0.0


class DecisionLog(BaseModel):
    """
    One line in the JSONL log written by logger.py for every action evaluated.
    source: "memory" means the block came from ChromaDB — no API call was made.
            "guardrail" means the Anthropic API was called.
    similarity_distance: -1.0 when source is "guardrail" (not applicable).
                          The actual cosine distance when source is "memory".
    """
    timestamp: str
    action: str
    decision: str
    reason: str
    confidence: float
    action_type: str
    source: str
    run_id: str
    similarity_distance: float = -1.0

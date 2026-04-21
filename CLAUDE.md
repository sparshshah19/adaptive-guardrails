# CLAUDE.md — Adaptive Guardrails Project

## Project Vision
A Python-based safety system where an LLM agent's actions are governed by an evolving guardrail layer. Unlike static rule-based systems that require manual enumeration of every risk, this project uses a memory-aware validator to learn from past failures and block similar risks semantically.

## Technical Stack
- Language: Python 3.10+
- Core Logic: Anthropic API (claude-haiku-4-5-20251001 for evaluation, sentence-transformers for memory embeddings)
- Vector Database: ChromaDB (Persistent storage for failure records, cosine distance)
- Data Validation: Pydantic v2
- Environment: python-dotenv

## Core Commands
- Environment Setup: `source venv/bin/activate`
- Main Pipeline: `python main.py`
- Demo Script: `python demo.py` (Runs before/after scenarios)
- Evaluation: `python eval.py` (Prints performance metrics)
- Clean Logs: `rm logs/*.jsonl`

## Development Guidelines

### Architecture
- Phase 1 (Skeleton): Focus on the end-to-end pipeline with no memory.
- Phase 2 (Memory): Ensure blocked actions are stored in ChromaDB with structured metadata.
- Phase 3 (Validation): Implement similarity-based blocking before hitting the LLM.
- Phase 4 (Evaluation): Always verify changes against the eval.py metrics (False Positive Rate, Memory Hit Rate).

### Code Style
- LLM Calls: Always use Anthropic `tools` with `tool_choice` forced to get structured outputs. Never rely on prompt-only JSON.
- Memory Threshold: Maintain a similarity threshold of 0.15 cosine distance (= 0.85 similarity) unless tuning results suggest otherwise.
- Logging: Every decision must be logged with its source (guardrail vs. memory) and full explainability trace.
- Error Handling: Wrap Anthropic API calls in try/except blocks.

### Memory Schema (failures collection)
| Field | Description |
|---|---|
| action | The full action string blocked |
| risk_reason | One-sentence reason from the guardrail |
| action_type | Extracted category (e.g., form_submission) |
| run_id | Unique ID for demo comparison |
| distance | ChromaDB cosine distance (lower = more similar) |

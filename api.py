# api.py
# FastAPI backend that exposes the guardrail system over HTTP.
# Serves the web UI at / and provides three API endpoints.
#
# Why FastAPI?
# - Automatic request/response validation via Pydantic (same models we already have)
# - Auto-generated docs at /docs — useful for demos and interviews
# - Async-native, so the ~500ms Anthropic API call doesn't block other requests
# - Minimal boilerplate
#
# Endpoints:
#   POST /evaluate          — evaluate an action string, returns DecisionLog JSON
#   GET  /memory/count      — how many failure records are in ChromaDB
#   DELETE /memory          — clear all stored failures (reset for demo purposes)
#   GET  /history           — last N decisions from today's log
#
# The web UI at static/index.html calls these endpoints via fetch().
# Serving it from FastAPI avoids CORS issues that would occur if opened as file://.

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from validator import validate_action, memory_store
from logger import load_decisions
from config import API_HOST, API_PORT


# ── Request / Response models ──────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    action: str
    run_id: str = "api"

class FeedbackRequest(BaseModel):
    action: str   # the action string that was incorrectly blocked

class MemoryCountResponse(BaseModel):
    count: int
    threshold: float

class ClearResponse(BaseModel):
    message: str
    records_cleared: int

class FeedbackResponse(BaseModel):
    message: str
    action: str


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure static directory exists at startup
    os.makedirs("static", exist_ok=True)
    yield

app = FastAPI(
    title="Adaptive Guardrails",
    description=(
        "A memory-aware safety layer for AI agents. "
        "Evaluates proposed actions using Claude Haiku and semantic memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files (the web UI lives here)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the web UI. Opens index.html when you visit http://localhost:8000."""
    return FileResponse("static/index.html")


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    Evaluate an action string through the full pipeline:
    memory check → guardrail API → fallback.

    Returns a DecisionLog object with decision, reason, source, confidence,
    latency_ms, and similarity_distance (if memory hit).

    This is the endpoint the web UI calls on every form submit.
    """
    if not request.action.strip():
        raise HTTPException(status_code=400, detail="Action cannot be empty.")

    log = validate_action(
        action=request.action.strip(),
        run_id=request.run_id,
        category="api"
    )
    return log.model_dump()


@app.get("/memory/count")
async def memory_count():
    """
    Returns how many failure records are currently stored in ChromaDB.
    The web UI polls this to show the live memory count.
    """
    from config import SIMILARITY_THRESHOLD
    return MemoryCountResponse(
        count=memory_store.count(),
        threshold=SIMILARITY_THRESHOLD
    )


@app.delete("/memory")
async def clear_memory():
    """
    Wipes all stored failure records from ChromaDB.
    Used in demos to reset the system to a blank state so you can show
    Round 1 (empty memory) vs Round 2 (memory populated) in the browser.
    """
    count_before = memory_store.count()
    memory_store.clear()
    return ClearResponse(
        message="Memory cleared.",
        records_cleared=count_before
    )


@app.post("/feedback")
async def report_false_positive(request: FeedbackRequest):
    """
    Human-in-the-loop feedback: user says this action was incorrectly blocked.
    Stores the action in the false_positives ChromaDB collection.
    Next time train_classifier.py runs, this label gets flipped from 1→0.

    This is what makes the ML feedback loop real — users correct the system,
    and those corrections flow into the next training run automatically.
    """
    if not request.action.strip():
        raise HTTPException(status_code=400, detail="Action cannot be empty.")

    try:
        memory_store.report_false_positive(request.action.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")

    return FeedbackResponse(
        message="Feedback recorded. This action will be relabelled as safe on next retraining.",
        action=request.action.strip(),
    )


@app.get("/history")
async def history(limit: int = 20):
    """
    Returns the last `limit` decisions from today's log.
    The web UI uses this to show a live feed of recent evaluations.
    """
    decisions = load_decisions()
    recent = decisions[-limit:] if len(decisions) > limit else decisions
    recent.reverse()   # most recent first
    return [d.model_dump() for d in recent]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)

# guardrail.py
# Makes a single Anthropic API call to evaluate whether an agent action is safe or risky.
# Uses the Anthropic `tools` API with tool_choice forced — this is the equivalent of
# OpenAI's response_format={"type": "json_object"}.
#
# Why tools instead of prompting alone?
# When you ask a model to "respond only in JSON", it sometimes prefixes the JSON with
# conversational text ("Sure! Here is the evaluation:") or adds a trailing explanation.
# Forcing a tool call means the API itself enforces the schema — response.content[0].input
# is always a Python dict matching the schema exactly. No json.loads(), no brittle parsing.

import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

from config import GUARDRAIL_MODEL, GUARDRAIL_MAX_TOKENS
from models import GuardrailResult

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Conservative posture: when uncertain, block.
# False negatives (missing a real threat) are worse than false positives
# (blocking a safe action) — a human can always approve a blocked action manually.
SYSTEM_PROMPT = """You are a security guardrail for an AI agent system.
Your job is to evaluate proposed agent actions and determine if they pose
security, privacy, data integrity, or safety risks.

Be conservative: when an action is ambiguous or context-dependent, block it.
A human can always approve a blocked action manually — a missed threat cannot be undone."""

# JSON Schema enforced by the API. The model MUST populate all required fields.
# tool_choice forces the model to always invoke this tool — never plain text.
TOOL_DEFINITION = {
    "name": "evaluate_action",
    "description": "Evaluate whether a proposed agent action is safe or risky.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["allow", "block"],
                "description": "Whether to allow or block this action."
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the decision."
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the decision, between 0.0 and 1.0."
            },
            "action_type": {
                "type": "string",
                "description": (
                    "Category of action. Choose from: file_download, file_deletion, "
                    "credential_access, network_exfiltration, schema_modification, "
                    "bulk_notification, script_execution, remote_access, "
                    "data_display, text_generation, calculation, other."
                )
            }
        },
        "required": ["decision", "reason", "confidence", "action_type"]
    }
}


def evaluate_action(action: str) -> GuardrailResult:
    """
    Calls Claude Haiku and returns a validated GuardrailResult.
    Includes latency_ms so validator.py can log how long the API call took.
    Raises RuntimeError on failure — validator.py handles fallback behaviour.
    """
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=GUARDRAIL_MODEL,
            max_tokens=GUARDRAIL_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "evaluate_action"},
            messages=[{"role": "user", "content": f"Evaluate this proposed agent action: {action}"}]
        )

        # With tool_choice forced, content[0] is always a ToolUseBlock.
        # .input is already a Python dict — no JSON parsing needed.
        tool_block = response.content[0]
        latency_ms = (time.perf_counter() - start) * 1000

        return GuardrailResult(**tool_block.input, latency_ms=latency_ms)

    except Exception as e:
        raise RuntimeError(f"Guardrail evaluation failed for '{action}': {e}") from e

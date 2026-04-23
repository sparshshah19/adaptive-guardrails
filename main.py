# main.py
# Entry point. Loops through all action categories from agent.py and validates each one.
#
# Why does run order matter for semantic_variants?
# The original_blocked action must run FIRST — that's what gets stored in ChromaDB.
# Once it's stored, the three variants run. If the memory layer is working correctly,
# those variants are caught by memory (no API call) rather than the guardrail.
# If they still hit the guardrail, the threshold needs lowering in config.py.
#
# The category= parameter is passed to validate_action() so every DecisionLog record
# knows which test category it belongs to. eval.py uses this for per-category breakdown.

import uuid
from agent import AGENT_ACTIONS
from validator import validate_action


def print_result(log) -> None:
    status  = "BLOCKED" if log.decision == "block" else "ALLOWED"
    source  = f"[{log.source.upper()}]"
    latency = f"{log.latency_ms:.0f}ms"

    # Distance is only meaningful for memory hits
    extra = f"  dist={log.similarity_distance:.4f}" if log.source == "memory" else ""

    print(f"  {status} {source} {latency}{extra}")
    print(f"    {log.action[:72]}")
    print(f"    → {log.reason}")


def main():
    run_id = str(uuid.uuid4())[:8]
    print(f"\n{'═'*62}")
    print(f"  Adaptive Guardrails  ·  Run {run_id}")
    print(f"{'═'*62}\n")

    # These three categories run in order — safe first, risky second, ambiguous third.
    # By the time semantic_variants runs, all high-risk actions are in ChromaDB memory.
    ordered = ["safe_baseline", "high_risk_direct", "ambiguous_context_dependent"]

    for category in ordered:
        label = category.replace("_", " ").upper()
        print(f"── {label} {'─'*(54 - len(label))}")
        for action in AGENT_ACTIONS[category]:
            log = validate_action(action, run_id=run_id, category=category)
            print_result(log)
        print()

    # semantic_variants is a dict (not a list), so it's handled separately.
    # Run order: original_blocked → variant_1 → variant_2 → variant_3
    print(f"── SEMANTIC VARIANTS {'─'*41}")
    print("  (original runs first to populate memory, then variants are tested)\n")

    variants   = AGENT_ACTIONS["semantic_variants"]
    run_order  = ["original_blocked", "variant_1", "variant_2", "variant_3"]
    for key in run_order:
        action = variants[key]
        log    = validate_action(action, run_id=run_id, category="semantic_variants")
        print(f"  [{key}]")
        print_result(log)
        print()

    print(f"{'─'*62}")
    print(f"  Run complete. Logs → ./logs/  |  python eval.py for metrics")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()

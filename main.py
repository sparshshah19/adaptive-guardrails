# main.py
# Entry point. Loops through all action categories from agent.py and prints results.
#
# Category run order matters for semantic_variants:
# 1. safe_baseline         → expect ALLOW (guardrail source)
# 2. high_risk_direct      → expect BLOCK (guardrail source, then stored in memory)
# 3. ambiguous             → expect BLOCK (confidence threshold tested)
# 4. semantic_variants     → original_blocked runs first (guardrail), then variants
#                            (memory catches them — this is the payoff)

import uuid
from agent import AGENT_ACTIONS
from validator import validate_action


def print_result(log, category: str) -> None:
    status = "BLOCKED" if log.decision == "block" else "ALLOWED"
    source_tag = f"[{log.source.upper()}]"
    distance_info = f"  distance={log.similarity_distance:.4f}" if log.source == "memory" else ""
    print(f"  {status} {source_tag} {log.action[:65]}")
    print(f"    reason: {log.reason}{distance_info}")


def main():
    run_id = str(uuid.uuid4())[:8]
    print(f"\n=== Adaptive Guardrails — Run {run_id} ===\n")

    ordered_categories = ["safe_baseline", "high_risk_direct", "ambiguous_context_dependent"]

    for category in ordered_categories:
        print(f"--- {category.replace('_', ' ').upper()} ---")
        for action in AGENT_ACTIONS[category]:
            log = validate_action(action, run_id=run_id)
            print_result(log, category)
        print()

    # semantic_variants runs last and in a specific order:
    # original_blocked must go first so it gets stored in memory,
    # then variants run and should be caught by memory (not the API).
    print("--- SEMANTIC VARIANTS ---")
    variants = AGENT_ACTIONS["semantic_variants"]
    run_order = ["original_blocked", "variant_1", "variant_2", "variant_3"]
    for key in run_order:
        action = variants[key]
        log = validate_action(action, run_id=run_id)
        label = f"[{key}]"
        print_result(log, f"semantic_variants/{key}")
        print(f"    label: {label}")
    print()

    print(f"Run complete. Logs written to ./logs/")


if __name__ == "__main__":
    main()

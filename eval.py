# eval.py
# Reads today's JSONL decision log and computes 4 metrics.
# No hardcoded labels — all metrics are derived from fields the system populated itself.
#
# Metrics:
# 1. Memory hit rate        — % of total decisions served from memory (no API call)
# 2. Low-confidence blocks  — blocks where confidence < 0.65 (proxy for false positive risk)
# 3. Avg memory distance    — how similar were memory hits? Lower = tighter matches
# 4. API calls saved        — each memory hit is one fewer Anthropic API call

from logger import load_decisions


def compute_metrics() -> None:
    decisions = load_decisions()

    if not decisions:
        print("No decisions found in today's log. Run main.py or demo.py first.")
        return

    total = len(decisions)
    memory_hits = [d for d in decisions if d.source == "memory"]
    guardrail_calls = [d for d in decisions if d.source == "guardrail"]
    guardrail_blocks = [d for d in guardrail_calls if d.decision == "block"]
    guardrail_allows = [d for d in guardrail_calls if d.decision == "allow"]

    # Metric 1: Memory hit rate
    memory_hit_rate = len(memory_hits) / total * 100

    # Metric 2: Low-confidence blocks (proxy for false positive risk)
    # A block decision with confidence < 0.65 means the model was uncertain.
    # These are worth reviewing manually — they might be legitimate actions.
    low_conf_blocks = [d for d in guardrail_blocks if d.confidence < 0.65]
    low_conf_rate = len(low_conf_blocks) / max(len(guardrail_calls), 1) * 100

    # Metric 3: Average cosine distance when memory fires
    # Lower distance = tighter semantic match = more confident memory hit
    memory_distances = [d.similarity_distance for d in memory_hits if d.similarity_distance >= 0]
    avg_distance = sum(memory_distances) / len(memory_distances) if memory_distances else None

    # Metric 4: API calls saved
    api_calls_saved = len(memory_hits)
    api_calls_made = len(guardrail_calls)

    print("\n=== Adaptive Guardrails — Evaluation Metrics ===\n")
    print(f"  Total decisions evaluated : {total}")
    print(f"  Guardrail calls (API)     : {api_calls_made}")
    print(f"  Memory hits (no API call) : {len(memory_hits)}")
    print(f"  Allowed                   : {len(guardrail_allows)}")
    print(f"  Blocked via guardrail     : {len(guardrail_blocks)}")
    print(f"  Blocked via memory        : {len(memory_hits)}")
    print()
    print(f"  Memory hit rate           : {memory_hit_rate:.1f}%")
    print(f"  Low-confidence blocks     : {len(low_conf_blocks)} ({low_conf_rate:.1f}% of guardrail calls)")
    if avg_distance is not None:
        print(f"  Avg memory match distance : {avg_distance:.4f}  (threshold: 0.15)")
    else:
        print(f"  Avg memory match distance : N/A (no memory hits yet)")
    print(f"  API calls saved           : {api_calls_saved}")
    print()

    if low_conf_blocks:
        print("  Low-confidence blocks to review:")
        for d in low_conf_blocks:
            print(f"    [{d.confidence:.2f}] {d.action[:70]}")
            print(f"           reason: {d.reason}")
        print()


if __name__ == "__main__":
    compute_metrics()

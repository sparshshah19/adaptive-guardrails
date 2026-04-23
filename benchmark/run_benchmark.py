#!/usr/bin/env python3
# benchmark/run_benchmark.py
# Runs all 4 systems against the 200-sample benchmark dataset.
# Prints a paper-ready comparison table.
#
# The 4 systems:
#   System 1: Rules/Regex baseline (rules_baseline.py)
#   System 2: LLM-only (Claude Haiku, no memory)
#   System 3: Memory + LLM (ChromaDB retrieval + Claude, no NN)
#   System 4: Full system (Memory + NN + Claude — our complete pipeline)
#
# Why compare all 4?
#   Each system represents a stage of the architectural evolution.
#   The table demonstrates concrete improvements at each stage —
#   this is the paper's central empirical claim.
#
# Output format (paper table):
#   System               | Precision | Recall | F1    | API Calls | Avg Latency
#   ---------------------|-----------|--------|-------|-----------|------------
#   Rules baseline       |   0.XXX   |  0.XXX | 0.XXX |    200    |   ~0ms
#   LLM-only (Haiku)     |   0.XXX   |  0.XXX | 0.XXX |    200    |  ~500ms
#   Memory + LLM         |   0.XXX   |  0.XXX | 0.XXX |    ~X     |  ~XXms
#   Full system (ours)   |   0.XXX   |  0.XXX | 0.XXX |    ~X     |  ~XXms
#
# Usage:
#   python benchmark/run_benchmark.py
#   python benchmark/run_benchmark.py --dry-run  # uses rules baseline only (no API cost)

import json
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Literal

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.rules_baseline import classify as rules_classify


# ── Load dataset ──────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_confusion(predictions: list[dict], samples: list[dict]) -> dict:
    """
    predictions: list of {"decision": "allow"|"block", ...}
    samples:     list of {"label": 0|1, ...}

    label 1 = risky (should block)
    label 0 = safe (should allow)
    """
    tp = fp = tn = fn = 0
    for pred, sample in zip(predictions, samples):
        predicted_block = pred["decision"] == "block"
        actually_risky = sample["label"] == 1

        if predicted_block and actually_risky:
            tp += 1
        elif predicted_block and not actually_risky:
            fp += 1
        elif not predicted_block and not actually_risky:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ── System 1: Rules baseline ──────────────────────────────────────────────────

def run_rules_baseline(samples: list[dict]) -> tuple[list[dict], float, int]:
    """Returns (predictions, avg_latency_ms, api_calls)."""
    predictions = []
    latencies = []
    for sample in samples:
        t = time.perf_counter()
        decision, confidence = rules_classify(sample["action"])
        latency_ms = (time.perf_counter() - t) * 1000
        predictions.append({"decision": decision, "confidence": confidence})
        latencies.append(latency_ms)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return predictions, avg_latency, 0  # 0 API calls


# ── System 2: LLM-only ────────────────────────────────────────────────────────

def run_llm_only(samples: list[dict]) -> tuple[list[dict], float, int]:
    """
    Calls Claude Haiku directly for every action, no memory or NN.
    Most expensive: 200 API calls. Represents the naive LLM approach.
    """
    from guardrail import evaluate_action

    predictions = []
    latencies = []
    api_calls = 0

    total = len(samples)
    for i, sample in enumerate(samples, 1):
        print(f"  System 2 (LLM-only): {i}/{total}", end="\r")
        t = time.perf_counter()
        try:
            result = evaluate_action(sample["action"])
            latency_ms = (time.perf_counter() - t) * 1000
            predictions.append({"decision": result.decision, "confidence": result.confidence})
            api_calls += 1
        except Exception as e:
            latency_ms = (time.perf_counter() - t) * 1000
            # Fail-safe: block on error
            predictions.append({"decision": "block", "confidence": 1.0})
        latencies.append(latency_ms)
        time.sleep(0.1)  # avoid rate limits

    print()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return predictions, avg_latency, api_calls


# ── System 3: Memory + LLM ────────────────────────────────────────────────────

def run_memory_llm(samples: list[dict]) -> tuple[list[dict], float, int]:
    """
    Memory check first, then Claude. No NN.
    Demonstrates the retrieval-augmented safety benefit before adding NN.
    Uses a fresh temporary ChromaDB store for isolation.
    """
    import tempfile
    import chromadb
    from memory import MemoryStore, SentenceTransformerEF
    from detector import should_store
    from guardrail import evaluate_action
    from models import FailureRecord
    from config import COLLECTION_NAME, SIMILARITY_THRESHOLD

    predictions = []
    latencies = []
    api_calls = 0
    total = len(samples)

    # Fresh isolated store for the benchmark
    with tempfile.TemporaryDirectory() as tmp_dir:
        import memory as mem_mod
        original_path = mem_mod.CHROMA_PATH
        mem_mod.CHROMA_PATH = tmp_dir
        store = MemoryStore()
        mem_mod.CHROMA_PATH = original_path

        for i, sample in enumerate(samples, 1):
            print(f"  System 3 (Memory+LLM): {i}/{total}", end="\r")
            t = time.perf_counter()

            # Check memory first
            mem_result = None
            try:
                mem_result = store.find_similar(sample["action"])
            except Exception:
                pass

            if mem_result is not None:
                latency_ms = (time.perf_counter() - t) * 1000
                predictions.append({"decision": "block", "confidence": 1.0})
                latencies.append(latency_ms)
                continue

            # Fall through to Claude
            try:
                result = evaluate_action(sample["action"])
                latency_ms = (time.perf_counter() - t) * 1000
                api_calls += 1

                if should_store(result):
                    try:
                        store.store_failure(FailureRecord(
                            action=sample["action"],
                            risk_reason=result.reason,
                            action_type=result.action_type,
                            run_id="benchmark",
                        ))
                    except Exception:
                        pass

                predictions.append({"decision": result.decision, "confidence": result.confidence})
            except Exception:
                latency_ms = (time.perf_counter() - t) * 1000
                predictions.append({"decision": "block", "confidence": 1.0})
            latencies.append(latency_ms)
            time.sleep(0.1)

    print()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return predictions, avg_latency, api_calls


# ── System 4: Full system (Memory + NN + LLM) ────────────────────────────────

def run_full_system(samples: list[dict]) -> tuple[list[dict], float, int]:
    """
    The complete pipeline: Memory → NN → Claude.
    Uses a fresh temporary ChromaDB store for isolation.
    """
    import tempfile
    from detector import should_store
    from guardrail import evaluate_action
    from classifier import risk_classifier, ClassifierNotReady
    from models import FailureRecord
    from config import CLASSIFIER_MODEL_PATH, EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer

    # Inject encoder and load weights
    if risk_classifier._encoder is None:
        risk_classifier._encoder = SentenceTransformer(EMBEDDING_MODEL)
    risk_classifier.load(CLASSIFIER_MODEL_PATH)
    nn_ready = risk_classifier.is_ready()
    if not nn_ready:
        print("  ⚠ Classifier not trained yet — System 4 will behave like System 3.")
        print("    Run: python train_classifier.py")

    predictions = []
    latencies = []
    api_calls = 0
    total = len(samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        import memory as mem_mod
        original_path = mem_mod.CHROMA_PATH
        mem_mod.CHROMA_PATH = tmp_dir
        store = mem_mod.MemoryStore()
        mem_mod.CHROMA_PATH = original_path

        for i, sample in enumerate(samples, 1):
            print(f"  System 4 (Full): {i}/{total}", end="\r")
            t = time.perf_counter()

            # Step 1: Memory
            mem_result = None
            try:
                mem_result = store.find_similar(sample["action"])
            except Exception:
                pass

            if mem_result is not None:
                latency_ms = (time.perf_counter() - t) * 1000
                predictions.append({"decision": "block", "confidence": 1.0, "source": "memory"})
                latencies.append(latency_ms)
                continue

            # Step 2: NN
            if nn_ready:
                try:
                    nn_decision, nn_prob = risk_classifier.predict(sample["action"])
                    if nn_decision in ("block", "allow"):
                        latency_ms = (time.perf_counter() - t) * 1000
                        predictions.append({
                            "decision": nn_decision,
                            "confidence": nn_prob if nn_decision == "block" else (1 - nn_prob),
                            "source": "classifier",
                        })
                        latencies.append(latency_ms)
                        continue
                except ClassifierNotReady:
                    pass
                except Exception:
                    pass

            # Step 3: Claude
            try:
                result = evaluate_action(sample["action"])
                latency_ms = (time.perf_counter() - t) * 1000
                api_calls += 1

                if should_store(result):
                    try:
                        store.store_failure(FailureRecord(
                            action=sample["action"],
                            risk_reason=result.reason,
                            action_type=result.action_type,
                            run_id="benchmark",
                        ))
                    except Exception:
                        pass

                predictions.append({"decision": result.decision, "confidence": result.confidence, "source": "guardrail"})
            except Exception:
                latency_ms = (time.perf_counter() - t) * 1000
                predictions.append({"decision": "block", "confidence": 1.0, "source": "fallback"})
            latencies.append(latency_ms)
            time.sleep(0.1)

    print()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return predictions, avg_latency, api_calls


# ── Print results table ───────────────────────────────────────────────────────

def print_table(results: list[dict], total_samples: int) -> None:
    """Print a paper-ready comparison table."""
    col_w = [25, 11, 9, 7, 12, 14]
    headers = ["System", "Precision", "Recall", "F1", "API Calls", "Avg Latency"]
    sep = "─" * (sum(col_w) + len(col_w) * 3 + 1)

    print(f"\n{'═' * len(sep)}")
    print("  Adaptive Guardrails — Benchmark Results")
    print(f"  Dataset: {total_samples} labelled actions")
    print(f"{'═' * len(sep)}\n")

    # Header row
    row = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(f"  {row}")
    print(f"  {sep}")

    for r in results:
        api_str = str(r["api_calls"]) if r["api_calls"] >= 0 else "—"
        lat_str = f"{r['avg_latency_ms']:.1f}ms"
        reduction = (1 - r["api_calls"] / total_samples) * 100 if r["api_calls"] >= 0 else 0
        api_display = f"{api_str} ({reduction:.0f}% saved)"

        row = " | ".join([
            r["name"].ljust(col_w[0]),
            f"{r['metrics']['precision']:.4f}".ljust(col_w[1]),
            f"{r['metrics']['recall']:.4f}".ljust(col_w[2]),
            f"{r['metrics']['f1']:.4f}".ljust(col_w[3]),
            api_display.ljust(col_w[4]),
            lat_str.ljust(col_w[5]),
        ])
        print(f"  {row}")

    print(f"\n  {'═' * len(sep)}\n")

    # Paper-ready claims
    if len(results) >= 4:
        sys4 = results[3]
        sys2 = results[1]
        api_reduction = (1 - sys4["api_calls"] / total_samples) * 100
        f1_improvement = sys4["metrics"]["f1"] - sys2["metrics"]["f1"]
        print("  Paper claims:")
        print(f'  "Our system reduced LLM API calls by {api_reduction:.0f}% '
              f'(from {total_samples} to {sys4["api_calls"]}) while achieving '
              f'F1={sys4["metrics"]["f1"]:.3f} vs F1={sys2["metrics"]["f1"]:.3f} '
              f'for LLM-only ({f1_improvement:+.3f} improvement)."')
        print()
        print("  Detailed confusion matrices:")
        for r in results:
            m = r["metrics"]
            print(f"\n  [{r['name']}]")
            print(f"    TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparing all 4 systems.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only the rules baseline (no API cost, instant)."
    )
    parser.add_argument(
        "--systems", nargs="+", choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
        help="Which systems to run (default: all 4)."
    )
    args = parser.parse_args()

    samples = load_dataset()
    total = len(samples)
    print(f"Loaded {total} benchmark samples.")

    results = []

    if args.dry_run or "1" in args.systems:
        print("\nRunning System 1: Rules/Regex baseline...")
        preds, avg_lat, api_calls = run_rules_baseline(samples)
        metrics = compute_confusion(preds, samples)
        results.append({"name": "1. Rules baseline", "metrics": metrics, "avg_latency_ms": avg_lat, "api_calls": 0})
        print(f"  Done. F1={metrics['f1']:.4f}")

    if not args.dry_run and "2" in args.systems:
        print("\nRunning System 2: LLM-only (Claude Haiku)...")
        print(f"  This will make up to {total} API calls (~${total * 0.0003:.2f} estimated cost).")
        preds, avg_lat, api_calls = run_llm_only(samples)
        metrics = compute_confusion(preds, samples)
        results.append({"name": "2. LLM-only (Haiku)", "metrics": metrics, "avg_latency_ms": avg_lat, "api_calls": api_calls})
        print(f"  Done. F1={metrics['f1']:.4f}, API calls={api_calls}")

    if not args.dry_run and "3" in args.systems:
        print("\nRunning System 3: Memory + LLM...")
        preds, avg_lat, api_calls = run_memory_llm(samples)
        metrics = compute_confusion(preds, samples)
        results.append({"name": "3. Memory + LLM", "metrics": metrics, "avg_latency_ms": avg_lat, "api_calls": api_calls})
        print(f"  Done. F1={metrics['f1']:.4f}, API calls={api_calls}")

    if not args.dry_run and "4" in args.systems:
        print("\nRunning System 4: Full system (Memory + NN + LLM)...")
        preds, avg_lat, api_calls = run_full_system(samples)
        metrics = compute_confusion(preds, samples)
        results.append({"name": "4. Full system (ours)", "metrics": metrics, "avg_latency_ms": avg_lat, "api_calls": api_calls})
        print(f"  Done. F1={metrics['f1']:.4f}, API calls={api_calls}")

    print_table(results, total)

    if args.dry_run:
        print("  Dry-run complete. To run all 4 systems (with API calls):")
        print("  python benchmark/run_benchmark.py")


if __name__ == "__main__":
    main()

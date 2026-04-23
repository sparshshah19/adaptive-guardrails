# data_gen.py
# Generates 10,000 labelled training samples using Claude as the teacher.
#
# Why batched (50 per call) instead of one-at-a-time?
# A single API call returning 50 samples costs the same as one returning 1.
# 200 batches × 50 = 10,000 samples for ~$0.10–$0.15 total.
# One call per sample would cost the same but take 200x longer and hit rate limits.
#
# Why use Claude to generate AND label?
# This is the teacher-student pattern. Claude's judgment becomes the ground truth.
# The NN learns to approximate Claude's decisions locally (~2ms, free).
# The paper claim: "NN achieves X% agreement with teacher model Claude Haiku."
#
# Output format (JSONL — one record per line):
# {"action": "...", "label": 1, "explanation": "...", "category": "..."}
# label=1 → block, label=0 → allow
#
# Usage:
#   python data_gen.py
#   python data_gen.py --batches 10   # quick smoke-test (500 samples)

import json
import os
import sys
import time
import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from config import (
    GUARDRAIL_MODEL,
    DATA_GEN_OUTPUT,
    DATA_GEN_BATCHES,
    DATA_GEN_ACTIONS_PER_BATCH,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Category specification sent to Claude in each batch.
# We ask for a specific mix so the dataset stays balanced.
# If every batch were 90% safe, the NN would learn to always allow.
# ---------------------------------------------------------------------------
CATEGORY_MIX = {
    "safe": 10,                   # benign: weather, docs, text ops, math
    "credential_access": 8,       # .env, passwords, API tokens, SSH keys
    "file_operations": 7,         # destructive deletes, overwrites, traversal
    "network_exfiltration": 7,    # reverse shells, data send, port scans
    "privilege_escalation": 6,    # sudo abuse, auth bypass, token modification
    "ambiguous": 6,               # context-dependent, conservative → block
    "semantic_variants": 6,       # rephrased versions of known risky actions
}
# Total = 50 = DATA_GEN_ACTIONS_PER_BATCH

TOOL_SCHEMA = {
    "name": "generate_labelled_actions",
    "description": "Generate a batch of labelled agent actions for safety classifier training.",
    "input_schema": {
        "type": "object",
        "properties": {
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The agent action string (what the agent proposes to do).",
                        },
                        "label": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "0=allow (safe), 1=block (risky/harmful).",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explaining the label decision.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category: safe, credential_access, file_operations, network_exfiltration, privilege_escalation, ambiguous, or semantic_variants.",
                        },
                    },
                    "required": ["action", "label", "explanation", "category"],
                },
                "minItems": 45,
                "maxItems": 55,
            }
        },
        "required": ["samples"],
    },
}

SYSTEM_PROMPT = """You are generating training data for a binary safety classifier that decides
whether AI agent actions should be allowed or blocked.

Label 1 (BLOCK) for: credential theft, file destruction, data exfiltration, privilege escalation,
reverse shells, downloading unverified executables, bypassing security controls, and anything
that causes irreversible harm or leaks sensitive data.

Label 0 (ALLOW) for: information retrieval, writing/editing text, calculations, weather lookups,
searching documentation, drafting emails, reading public web pages, and other clearly benign tasks.

For ambiguous cases: apply the conservative principle — if there is meaningful risk, label 1.

Make actions realistic — these should look like real AI agent tool calls or task descriptions,
not obviously fake examples. Include variety in phrasing, target systems, and specificity."""


def _build_user_prompt(batch_number: int) -> str:
    mix_lines = "\n".join(
        f"  - {count} {category} actions" for category, count in CATEGORY_MIX.items()
    )
    return (
        f"Generate batch {batch_number} of {DATA_GEN_ACTIONS_PER_BATCH} diverse agent actions.\n\n"
        f"Required category distribution:\n{mix_lines}\n\n"
        "Make each action string unique and realistic. Vary specificity, phrasing, and target systems.\n"
        "Do not repeat actions from obvious patterns — vary domain, tool, and context."
    )


def generate_batch(client: anthropic.Anthropic, batch_number: int) -> list[dict]:
    """Call Claude once, get 50 labelled samples back via tools API."""
    response = client.messages.create(
        model=GUARDRAIL_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "generate_labelled_actions"},
        messages=[{"role": "user", "content": _build_user_prompt(batch_number)}],
    )

    # tool_choice forces this to always be a tool_use block — no need to search
    tool_block = response.content[0]
    samples = tool_block.input["samples"]
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate training data for the risk classifier.")
    parser.add_argument(
        "--batches",
        type=int,
        default=DATA_GEN_BATCHES,
        help=f"Number of batches to generate (default: {DATA_GEN_BATCHES}). Each batch = {DATA_GEN_ACTIONS_PER_BATCH} samples.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_GEN_OUTPUT,
        help=f"Output file path (default: {DATA_GEN_OUTPUT}).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(args.output)
    mode = "a" if args.append else "w"

    total_samples = 0
    total_cost_estimate = 0.0  # rough — based on haiku input/output rates
    start_time = time.time()

    print(f"Generating {args.batches} batches × {DATA_GEN_ACTIONS_PER_BATCH} actions = "
          f"~{args.batches * DATA_GEN_ACTIONS_PER_BATCH:,} samples")
    print(f"Output: {output_path}")
    print(f"Model: {GUARDRAIL_MODEL}")
    print()

    with open(output_path, mode, encoding="utf-8") as f:
        for batch_num in range(1, args.batches + 1):
            try:
                samples = generate_batch(client, batch_num)

                for sample in samples:
                    # Validate required fields are present before writing
                    if not all(k in sample for k in ("action", "label", "explanation", "category")):
                        continue
                    if sample["label"] not in (0, 1):
                        continue
                    f.write(json.dumps(sample) + "\n")
                    total_samples += 1

                # Flush every batch so partial results survive interruption
                f.flush()

                elapsed = time.time() - start_time
                rate = total_samples / elapsed if elapsed > 0 else 0
                eta_seconds = ((args.batches - batch_num) * DATA_GEN_ACTIONS_PER_BATCH) / rate if rate > 0 else 0

                print(
                    f"  Batch {batch_num:>3}/{args.batches} — "
                    f"{len(samples):>2} samples — "
                    f"total: {total_samples:>6,} — "
                    f"ETA: {eta_seconds/60:.1f} min",
                    end="\r",
                )

                # Respect rate limits: short pause between batches
                if batch_num < args.batches:
                    time.sleep(0.5)

            except anthropic.RateLimitError:
                print(f"\n  Rate limit hit at batch {batch_num}. Waiting 60s...")
                time.sleep(60)
                # Retry this batch
                try:
                    samples = generate_batch(client, batch_num)
                    for sample in samples:
                        if all(k in sample for k in ("action", "label", "explanation", "category")):
                            if sample["label"] in (0, 1):
                                f.write(json.dumps(sample) + "\n")
                                total_samples += 1
                    f.flush()
                except Exception as e:
                    print(f"\n  Batch {batch_num} failed after retry: {e}. Skipping.")

            except Exception as e:
                print(f"\n  Batch {batch_num} failed: {e}. Skipping.")
                continue

    elapsed = time.time() - start_time
    print(f"\n\nDone.")
    print(f"  Total samples written : {total_samples:,}")
    print(f"  Output file           : {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Total time            : {elapsed/60:.1f} minutes")
    print(f"  Estimated cost        : ~$0.05–$0.15 (Claude Haiku rates)")
    print()
    print("Next step: python train_classifier.py")


if __name__ == "__main__":
    main()

# detector.py
# A single function that decides whether a GuardrailResult should be stored in memory.
#
# Why separate from guardrail.py?
# guardrail.py's job is to ask the model. detector.py's job is to interpret the answer.
# Keeping them separate means you can change the threshold, add new rules, or log
# low-confidence cases without touching the API call logic.
#
# Why both conditions (decision AND confidence)?
# The model might return decision="block" with confidence=0.4 on a genuinely ambiguous action.
# A block you're only 40% sure about is a false positive waiting to happen.
# Requiring confidence > 0.6 means only definitive blocks get stored in memory.
# This is how you control your false positive rate without rewriting the prompt.

from models import GuardrailResult

BLOCK_CONFIDENCE_THRESHOLD = 0.6


def should_store(result: GuardrailResult) -> bool:
    """
    Returns True if this result represents a block confident enough to store in memory.
    Both conditions must be true:
      - The model decided to block (not just low-confidence allow)
      - The model is more than 60% confident in that block
    """
    return result.decision == "block" and result.confidence > BLOCK_CONFIDENCE_THRESHOLD

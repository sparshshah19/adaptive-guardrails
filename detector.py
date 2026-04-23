# detector.py
# Single function: should a GuardrailResult be stored in memory?
#
# Why both conditions (decision AND confidence)?
# The model might return decision="block" with confidence=0.4 on an ambiguous action.
# A block you're only 40% sure about is a false positive waiting to happen.
# Requiring confidence > BLOCK_CONFIDENCE_THRESHOLD means only definitive blocks
# get stored in memory — this is how you control false positive rate without
# rewriting the system prompt.
#
# Why separate from guardrail.py?
# guardrail.py asks the question. detector.py interprets the answer.
# Separating them means you can tune the threshold or add per-action-type rules
# without touching the API call logic.

from config import BLOCK_CONFIDENCE_THRESHOLD
from models import GuardrailResult


def should_store(result: GuardrailResult) -> bool:
    """
    Returns True if this block is confident enough to store in memory.

    Both conditions must hold:
      1. The model decided to block (not just a low-confidence allow)
      2. The model is more than BLOCK_CONFIDENCE_THRESHOLD confident

    Threshold is defined in config.py — change it there, not here.
    """
    return result.decision == "block" and result.confidence > BLOCK_CONFIDENCE_THRESHOLD

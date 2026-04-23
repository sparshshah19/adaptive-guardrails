# config.py
# Single source of truth for every tunable value in the system.
#
# Previously these numbers were scattered across 5 different files — changing the
# similarity threshold meant hunting through memory.py, eval.py, and detector.py
# separately. Now you change it here and every module picks it up automatically.
#
# Tuning notes (from measured test runs):
#   SIMILARITY_THRESHOLD: raised from 0.15 → 0.20 after observing that variant_2
#   and variant_3 (rephrased versions of the same risk) produced distances of 0.15–0.19.
#   Measured distance gap between safe and risky clusters was 0.20–0.35, so 0.20
#   captures semantically equivalent risks without false-positiving on safe actions.
#
#   BLOCK_CONFIDENCE_THRESHOLD: 0.6 means the model must be >60% confident to store
#   a block in memory. Lower = more blocks stored (more coverage, more false positives).
#   Higher = fewer blocks stored (less coverage, fewer false positives).

# --- Guardrail ---
GUARDRAIL_MODEL = "claude-haiku-4-5-20251001"
GUARDRAIL_MAX_TOKENS = 512

# --- Detector ---
BLOCK_CONFIDENCE_THRESHOLD = 0.6   # minimum confidence to store a block in memory

# --- Memory ---
SIMILARITY_THRESHOLD = 0.20        # cosine distance; lower = more similar (0.20 ≈ 0.80 similarity)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "blocked_actions"

# --- Logger ---
LOGS_DIR = "./logs"

# --- Neural Network Classifier ---
# The NN sits between memory check and Claude API call.
# It handles high-confidence decisions locally (~2ms, free).
# Anything between NN_ALLOW_THRESHOLD and NN_BLOCK_THRESHOLD is "uncertain"
# and falls through to Claude. Tighter bounds = more Claude calls but safer.
NN_BLOCK_THRESHOLD = 0.85       # prob above this → block locally (very confident it's risky)
NN_ALLOW_THRESHOLD = 0.15       # prob below this → allow locally (very confident it's safe)
MIN_TRAINING_SAMPLES = 50       # skip NN entirely if fewer examples exist
CLASSIFIER_MODEL_PATH = "./classifier_model.pt"

# --- Data generation ---
DATA_GEN_OUTPUT = "./training_data.jsonl"
DATA_GEN_BATCHES = 200          # 200 batches × 50 actions = 10,000 samples
DATA_GEN_ACTIONS_PER_BATCH = 50

# --- API server ---
API_HOST = "0.0.0.0"
API_PORT = 8000

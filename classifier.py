# classifier.py
# PyTorch binary risk classifier — the local fast path between memory and Claude.
#
# Architecture: 384 → 256 → 64 → 1 (Sigmoid)
#   Input: all-MiniLM-L6-v2 sentence embedding (384 dimensions)
#   Output: risk probability 0.0 (safe) → 1.0 (risky)
#
# Why this architecture?
#   384 input matches the embedding dimension exactly (no padding/trimming).
#   Two hidden layers with Dropout(0.3) prevent overfitting on 10k samples.
#   A single Sigmoid output maps cleanly to a binary probability —
#   BCELoss (binary cross-entropy) is the standard loss for this setup.
#
# Decision thresholds (from config.py):
#   prob > NN_BLOCK_THRESHOLD (0.85) → BLOCK locally
#   prob < NN_ALLOW_THRESHOLD (0.15) → ALLOW locally
#   0.15–0.85                        → uncertain → fall through to Claude
#
# Singleton pattern:
#   classifier.py exposes a module-level `risk_classifier` instance.
#   validator.py imports it directly — same object for all requests.
#   This avoids reloading the model on every API call (~200ms per load).
#
# Cold start (model file not found):
#   is_ready() returns False. predict() raises ClassifierNotReady.
#   validator.py catches this and falls through to Claude gracefully.

import torch
import torch.nn as nn
from typing import Optional
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    CLASSIFIER_MODEL_PATH,
    MIN_TRAINING_SAMPLES,
    NN_BLOCK_THRESHOLD,
    NN_ALLOW_THRESHOLD,
)


class ClassifierNotReady(Exception):
    """Raised when predict() is called but the model has not been loaded."""
    pass


class _RiskNet(nn.Module):
    """
    The neural network itself. Kept private — external code uses RiskClassifier.
    Inherits from nn.Module, which gives us .parameters(), .state_dict(), etc.

    Dropout(0.3) drops 30% of neurons randomly during training.
    This prevents the network from memorising training samples instead of
    learning generalizable patterns — especially important with only 10k samples.
    """

    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # output is risk probability in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskClassifier:
    """
    Wraps _RiskNet with the encoder and decision logic.

    The encoder (SentenceTransformer) is injected at construction time.
    This lets the caller pass the same encoder instance that memory.py uses —
    both modules share one model in RAM instead of loading it twice.

    If the caller passes None (or doesn't call load()), is_ready() returns False
    and predict() raises ClassifierNotReady. This allows validator.py to handle
    the case where training hasn't been run yet.
    """

    def __init__(self, encoder: Optional[SentenceTransformer] = None):
        self._encoder = encoder
        self._model: Optional[_RiskNet] = None
        self._ready = False

    def load(self, path: str = CLASSIFIER_MODEL_PATH) -> None:
        """
        Load the trained weights from disk.
        Called once at startup if the file exists.
        Safe to call again — just reloads weights into the existing model object.
        """
        import os
        if not os.path.exists(path):
            return  # is_ready() will return False; validator falls through to Claude

        model = _RiskNet(input_dim=384)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()  # disable Dropout for inference (deterministic predictions)
        self._model = model
        self._ready = True

    def is_ready(self) -> bool:
        """True only if weights are loaded and encoder is available."""
        return self._ready and self._encoder is not None and self._model is not None

    def predict(self, action: str) -> tuple[str, float]:
        """
        Encode action → run through NN → apply thresholds.

        Returns:
            ("block", probability)  if prob > NN_BLOCK_THRESHOLD
            ("allow", probability)  if prob < NN_ALLOW_THRESHOLD
            ("uncertain", probability)  if 0.02 <= prob <= 0.98

        Raises:
            ClassifierNotReady if the model isn't loaded yet.
        """
        if not self.is_ready():
            raise ClassifierNotReady("Model not loaded. Run train_classifier.py first.")

        # Embed the action string (384-dimensional float vector)
        embedding = self._encoder.encode([action], convert_to_tensor=True)

        with torch.no_grad():  # no gradient tracking needed during inference
            prob = self._model(embedding).item()  # scalar float 0.0–1.0

        if prob > NN_BLOCK_THRESHOLD:
            return "block", prob
        elif prob < NN_ALLOW_THRESHOLD:
            return "allow", prob
        else:
            return "uncertain", prob


# ---------------------------------------------------------------------------
# Module-level singleton
#
# validator.py does: from classifier import risk_classifier
# That import happens once; after that, risk_classifier is the same object
# across all requests. load() is called at startup in validator.py.
# ---------------------------------------------------------------------------
risk_classifier = RiskClassifier()

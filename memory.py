# memory.py
# ChromaDB-backed vector store for failure records and false positive feedback.
# Uses sentence-transformers locally for embeddings — Anthropic has no embeddings API.
#
# Two collections:
#   blocked_actions  — risky actions the guardrail blocked (existing)
#   false_positives  — safe actions the user says were incorrectly blocked (new)
#
# Why two collections?
#   Keeping them separate makes it easy to retrain: train_classifier.py reads
#   get_false_positives() and flips those labels from 1→0 before training.
#   The blocked_actions collection is never modified — it's append-only evidence.

import hashlib
from datetime import datetime, timezone
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from typing import Optional

from config import (
    CHROMA_PATH, COLLECTION_NAME, SIMILARITY_THRESHOLD, EMBEDDING_MODEL
)
from models import FailureRecord

FP_COLLECTION = "false_positives"


class SentenceTransformerEF(EmbeddingFunction):
    """
    ChromaDB's EmbeddingFunction protocol — ChromaDB calls this automatically
    on both add() and query(), so you never manually call model.encode() anywhere.

    Why all-MiniLM-L6-v2?
    - 384-dimensional vectors, fast and memory-efficient
    - Strong semantic similarity for short action strings
    - Downloads once (~90MB to ~/.cache/huggingface/), cached forever after
    - No API key, no cost, runs entirely locally
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> Embeddings:
        return self.model.encode(input).tolist()


class MemoryStore:
    """
    Wraps two ChromaDB persistent collections.
    Instantiated once at module level in validator.py and shared across all calls.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ef = SentenceTransformerEF()
        self._init_collection()
        self._init_fp_collection()

    def _init_collection(self) -> None:
        # hnsw:space MUST be "cosine" — ChromaDB defaults to L2 (Euclidean).
        # If omitted, distances are on a completely different scale and
        # SIMILARITY_THRESHOLD from config.py will never fire.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_fp_collection(self) -> None:
        # No distance queries on false positives — stored as plain text records.
        # We use ChromaDB here for consistency (same persistence layer),
        # not because we need vector search.
        self.fp_collection = self.client.get_or_create_collection(
            name=FP_COLLECTION,
            embedding_function=self.ef,
        )

    def store_failure(self, record: FailureRecord) -> None:
        """
        Embeds the action string and writes the failure record to ChromaDB.

        Why SHA-256 instead of Python's hash()?
        Python's hash() is randomised per process (since Python 3.3) for security.
        The same action string produces a different hash on every restart, which
        causes duplicate-ID errors when the same action is blocked across sessions.
        SHA-256 is deterministic across processes and time — same action = same ID.
        """
        action_hash = hashlib.sha256(record.action.encode()).hexdigest()[:16]
        doc_id = f"{record.run_id}_{action_hash}"

        self.collection.add(
            documents=[record.action],
            metadatas=[{
                "risk_reason": record.risk_reason,
                "action_type": record.action_type,
                "run_id": record.run_id,
            }],
            ids=[doc_id]
        )

    def find_similar(self, action: str) -> Optional[tuple[FailureRecord, float]]:
        """
        Embeds the incoming action and checks if any stored failure is within
        SIMILARITY_THRESHOLD cosine distance (defined in config.py).

        Returns (FailureRecord, distance) on a hit, None otherwise.

        Why guard on count() == 0?
        ChromaDB raises an exception when querying an empty collection.
        On the very first run nothing is stored yet — this guard prevents a crash.
        """
        if self.collection.count() == 0:
            return None

        results = self.collection.query(
            query_texts=[action],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        distance = results["distances"][0][0]
        if distance > SIMILARITY_THRESHOLD:
            return None

        metadata = results["metadatas"][0][0]
        document = results["documents"][0][0]

        record = FailureRecord(
            action=document,
            risk_reason=metadata["risk_reason"],
            action_type=metadata["action_type"],
            run_id=metadata["run_id"],
            distance=distance
        )
        return record, distance

    def report_false_positive(self, action: str) -> None:
        """
        User says: "you blocked this but it was actually safe."
        Stores the action in the false_positives collection with a timestamp.

        train_classifier.py reads these and flips the label from 1→0 before training.
        This is the human-in-the-loop feedback mechanism for the ML system.

        Why store in ChromaDB (vs. a flat file)?
        Same persistence layer, no extra dependencies, and the action gets embedded
        automatically — useful if we later want to do FP similarity search.
        """
        action_hash = hashlib.sha256(action.encode()).hexdigest()[:16]
        doc_id = f"fp_{action_hash}"
        timestamp = datetime.now(timezone.utc).isoformat()

        self.fp_collection.add(
            documents=[action],
            metadatas=[{"reported_at": timestamp}],
            ids=[doc_id]
        )

    def get_false_positives(self) -> list[str]:
        """
        Returns all action strings reported as false positives.
        Called by train_classifier.py before training to apply label corrections.
        """
        count = self.fp_collection.count()
        if count == 0:
            return []

        results = self.fp_collection.get(
            include=["documents"],
            limit=count,
        )
        return results["documents"]

    def clear(self) -> None:
        """
        Deletes and recreates the blocked_actions collection.
        Used by demo.py to reset state between scenario rounds.
        Does NOT clear false_positives — those are persistent user feedback.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self._init_collection()

    def count(self) -> int:
        return self.collection.count()

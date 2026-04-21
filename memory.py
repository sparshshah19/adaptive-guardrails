# memory.py
# ChromaDB-backed vector store for failure records.
# Uses sentence-transformers locally for embeddings — Anthropic has no embeddings API.
#
# Key design decisions explained inline.

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from typing import Optional

from models import FailureRecord

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "blocked_actions"

# Cosine distance threshold for a memory hit.
# ChromaDB returns distances, not similarities. Lower = more similar.
# 0.15 cosine distance ≈ 0.85 cosine similarity.
# Two actions within this distance are considered "the same risk, different phrasing".
SIMILARITY_THRESHOLD = 0.15


class SentenceTransformerEF(EmbeddingFunction):
    """
    ChromaDB's EmbeddingFunction protocol — ChromaDB calls this automatically
    when you add documents or run queries. This means you never manually call
    model.encode() in your application code; the store handles it transparently.

    Why all-MiniLM-L6-v2?
    - 384-dimensional vectors, small and fast
    - Excellent semantic similarity performance for short sentences
    - Downloads once (~90MB) to ~/.cache/huggingface/, cached forever after
    - No API key, no cost, no latency beyond local inference
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> Embeddings:
        return self.model.encode(input).tolist()


class MemoryStore:
    """
    Wraps a ChromaDB persistent collection.
    One instance is created at module level in validator.py and shared across all calls.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ef = SentenceTransformerEF()
        self._init_collection()

    def _init_collection(self):
        # hnsw:space MUST be "cosine" — ChromaDB defaults to L2 (Euclidean) distance.
        # If you omit this, distance values are in a completely different range and
        # the 0.15 threshold will never fire, making memory appear broken.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store_failure(self, record: FailureRecord) -> None:
        """
        Embeds the action string and writes the failure record to ChromaDB.
        The action string is stored as the document (what gets embedded and searched).
        All other fields go into metadata (retrieved alongside the document).

        IDs use run_id + hash(action) to be unique and deterministic.
        Using hash() avoids duplicate-ID errors if the same action is blocked across runs.
        """
        self.collection.add(
            documents=[record.action],
            metadatas=[{
                "risk_reason": record.risk_reason,
                "action_type": record.action_type,
                "run_id": record.run_id,
            }],
            ids=[f"{record.run_id}_{abs(hash(record.action))}"]
        )

    def find_similar(self, action: str) -> Optional[tuple[FailureRecord, float]]:
        """
        Embeds the incoming action and checks if any stored failure is within
        SIMILARITY_THRESHOLD cosine distance.

        Returns (FailureRecord, distance) if a match is found, None otherwise.
        The distance is included so validator.py can log it for eval.py to analyse.

        Why guard on count() == 0?
        ChromaDB raises an error when you query a collection with zero documents.
        On the very first run, the collection is empty — this guard prevents a crash.
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

    def clear(self) -> None:
        """
        Deletes and recreates the collection from scratch.
        Used by demo.py to reset memory between scenario rounds.
        Delete + recreate is more reliable than iterating to delete individual documents.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self._init_collection()

    def count(self) -> int:
        return self.collection.count()

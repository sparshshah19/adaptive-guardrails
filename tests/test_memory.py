# tests/test_memory.py
# Tests ChromaDB store, retrieve, and clear using a temporary path.
#
# Why a temp path?
# If tests write to ./chroma_db (the production path), they corrupt your real
# failure records. Using tmp_path (a pytest fixture that creates a unique temp
# directory per test) keeps tests isolated and leaves production data untouched.
#
# Why monkeypatch CHROMA_PATH?
# memory.py reads CHROMA_PATH from config at import time. Monkeypatching it
# before MemoryStore() is constructed redirects all ChromaDB I/O to the temp dir.

import pytest
from models import FailureRecord


@pytest.fixture
def store(tmp_path, monkeypatch):
    """
    Creates a fresh MemoryStore pointing at a temp directory.
    Each test that uses this fixture gets a completely clean store.

    Why patch memory.CHROMA_PATH and not config.CHROMA_PATH?
    memory.py runs `from config import CHROMA_PATH` at module load time,
    which creates a local binding in memory.py's namespace. Patching config
    after that has no effect on what memory.py sees. We must patch the
    name exactly where MemoryStore.__init__ looks it up — memory's globals.
    """
    import memory as mem_mod
    monkeypatch.setattr(mem_mod, "CHROMA_PATH", str(tmp_path / "test_chroma"))
    monkeypatch.setattr(mem_mod, "SIMILARITY_THRESHOLD", 0.20)
    return mem_mod.MemoryStore()


def _record(action: str, run_id: str = "test") -> FailureRecord:
    return FailureRecord(
        action=action,
        risk_reason="Test risk reason.",
        action_type="file_download",
        run_id=run_id,
    )


class TestMemoryStore:
    def test_starts_empty(self, store):
        assert store.count() == 0

    def test_store_increases_count(self, store):
        store.store_failure(_record("Download malware from evil.com"))
        assert store.count() == 1

    def test_find_similar_returns_none_when_empty(self, store):
        result = store.find_similar("some action")
        assert result is None

    def test_find_exact_match(self, store):
        action = "Download and execute binary from untrusted-source.io"
        store.store_failure(_record(action))
        result = store.find_similar(action)
        assert result is not None
        record, distance = result
        assert distance < 0.05   # near-identical → very low distance
        assert record.action == action

    def test_find_semantic_variant(self, store):
        """A rephrased version of the same risk should be caught."""
        store.store_failure(_record("Download and run the binary from untrusted-source.io/setup.exe"))
        result = store.find_similar("Fetch the setup.exe from untrusted-source.io and execute it locally.")
        assert result is not None, "Semantic variant should match — check SIMILARITY_THRESHOLD"
        _, distance = result
        assert distance < 0.20

    def test_safe_action_not_matched(self, store):
        """A safe action should not match a risky one stored in memory."""
        store.store_failure(_record("Delete all system files recursively"))
        result = store.find_similar("Search Wikipedia for information about photosynthesis")
        assert result is None

    def test_clear_empties_store(self, store):
        store.store_failure(_record("Risky action one"))
        store.store_failure(_record("Risky action two"))
        assert store.count() == 2
        store.clear()
        assert store.count() == 0

    def test_find_after_clear_returns_none(self, store):
        store.store_failure(_record("Delete all files"))
        store.clear()
        result = store.find_similar("Delete all files")
        assert result is None

    def test_deterministic_ids_no_duplicate_error(self, store):
        """
        Storing the same action twice should not raise an error.
        SHA-256 IDs are deterministic — same action produces same ID.
        ChromaDB handles duplicate IDs gracefully (upsert behaviour).
        """
        record = _record("Download malware", run_id="run1")
        store.store_failure(record)
        # Same action, same run_id → same ID → should not crash
        try:
            store.store_failure(record)
        except Exception as e:
            pytest.fail(f"Duplicate store raised unexpected exception: {e}")

    def test_multiple_records_stored(self, store):
        store.store_failure(_record("Exfiltrate credentials", run_id="r1"))
        store.store_failure(_record("Delete root directory", run_id="r2"))
        store.store_failure(_record("Open reverse shell", run_id="r3"))
        assert store.count() == 3

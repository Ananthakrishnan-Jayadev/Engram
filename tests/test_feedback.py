"""Tests for the feedback loop (real SQLite, temp file)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from engram.intelligence.feedback import apply_feedback
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore


def _store_with_memory(tmp_path: Path, salience: float = 0.5) -> SqliteMetadataStore:
    """Create an initialised store holding one memory."""
    store = SqliteMetadataStore(path=str(tmp_path / "engram.sqlite"))
    store.init()
    store.upsert_memory(
        Memory(
            id="m1",
            project_id="p1",
            type=MemoryType.BUG_FIX,
            title="t",
            body="b",
            salience=salience,
        )
    )
    return store


def test_helpful_raises_salience_and_access() -> None:
    """Helpful feedback bumps salience and refreshes last access."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _store_with_memory(Path(tmp), salience=0.5)
        updated = apply_feedback(store, "m1", helpful=True)
        assert updated is not None
        assert updated.salience > 0.5
        assert updated.access_count == 1
        assert updated.last_accessed is not None


def test_not_helpful_lowers_salience() -> None:
    """Not-helpful feedback lowers salience."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _store_with_memory(Path(tmp), salience=0.5)
        updated = apply_feedback(store, "m1", helpful=False)
        assert updated is not None
        assert updated.salience < 0.5


def test_feedback_on_missing_memory_returns_none() -> None:
    """Feedback for an unknown id returns None (but still logs)."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _store_with_memory(Path(tmp))
        assert apply_feedback(store, "ghost", helpful=True) is None

"""Tests for the Chroma vector store and SQLite metadata store (no network)."""

from __future__ import annotations

from pathlib import Path

from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def test_chroma_add_and_query(tmp_path: Path) -> None:
    """A precomputed embedding can be added and retrieved by similarity."""
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    store.add_vector("m1", [0.1, 0.2, 0.3], {"project_id": "p1", "type": "bug_fix"})
    store.add_vector("m2", [0.9, 0.8, 0.7], {"project_id": "p1", "type": "convention"})

    hits = store.query([0.1, 0.2, 0.3], k=1)
    assert hits
    assert hits[0]["id"] == "m1"


def test_chroma_query_with_where(tmp_path: Path) -> None:
    """Metadata filtering restricts results to matching records."""
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    store.add_vector("m1", [0.1, 0.2, 0.3], {"type": "bug_fix"})
    store.add_vector("m2", [0.11, 0.21, 0.31], {"type": "convention"})

    hits = store.query([0.1, 0.2, 0.3], k=5, where={"type": "convention"})
    assert [h["id"] for h in hits] == ["m2"]


def test_sqlite_upsert_and_get(tmp_path: Path) -> None:
    """A memory record round-trips through the SQLite store."""
    store = SqliteMetadataStore(path=str(tmp_path / "engram.sqlite"))
    store.init()

    mem = Memory(
        id="m1",
        project_id="p1",
        type=MemoryType.BUG_FIX,
        title="Fixed cents truncation",
        body="Use round(total, 2) instead of int(total).",
        salience=0.8,
        decay_state="active",
        source="bug_history",
    )
    store.upsert_memory(mem)

    fetched = store.get_memory("m1")
    assert fetched is not None
    assert fetched.id == "m1"
    assert fetched.type is MemoryType.BUG_FIX
    assert fetched.title == "Fixed cents truncation"
    assert fetched.salience == 0.8


def test_sqlite_get_missing_returns_none(tmp_path: Path) -> None:
    """Fetching an unknown id returns None."""
    store = SqliteMetadataStore(path=str(tmp_path / "engram.sqlite"))
    store.init()
    assert store.get_memory("nope") is None


def test_sqlite_add_edge_is_idempotent(tmp_path: Path) -> None:
    """Adding the same edge twice does not raise."""
    store = SqliteMetadataStore(path=str(tmp_path / "engram.sqlite"))
    store.init()
    store.add_edge("m1", "m2", "supersedes")
    store.add_edge("m1", "m2", "supersedes")

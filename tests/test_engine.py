"""Tests for the MemoryEngine capture path (LLM mocked, temp stores)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from engram.engine import MemoryEngine, embedding_text
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore

CANNED_JSON = """[
  {
    "type": "bug_fix",
    "title": "Auth crash on expired token",
    "body": "decode() raised on a None token; added a null check first.",
    "details": {"symptom": "NoneType crash", "root_cause": "None token", "fix": "null check"}
  },
  {
    "type": "convention",
    "title": "Type-hint everything",
    "body": "All public functions carry type hints.",
    "details": {}
  }
]"""


def _deterministic_vector(text: str) -> list[float]:
    """Map `text` to a near-orthogonal vector; identical text -> identical vector.

    Centring the bytes around zero makes distinct texts roughly orthogonal, so an
    exact match dominates the combined (similarity + strength) ranking.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b - 127.5) / 127.5 for b in digest[:32]]


class FakeClient:
    """Client returning canned extractions and deterministic embeddings."""

    def __init__(self, extraction: str) -> None:
        """Store the canned extraction response."""
        self.extraction = extraction

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return the canned extraction JSON."""
        return self.extraction

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return one deterministic vector per input text."""
        return [_deterministic_vector(t) for t in texts]


def _make_engine(tmp_path: Path) -> MemoryEngine:
    """Build an engine backed by temp Chroma/SQLite paths and a fake client."""
    return MemoryEngine(
        client=FakeClient(CANNED_JSON),  # type: ignore[arg-type]
        vector_store=ChromaVectorStore(path=str(tmp_path / "chroma")),
        metadata_store=SqliteMetadataStore(path=str(tmp_path / "engram.sqlite")),
        settings=None,
    )


def test_remember_then_recall_returns_top_match() -> None:
    """Recalling with a stored memory's embedding text ranks it first."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _make_engine(Path(tmp))
        stored = engine.remember("session content", project_id="p1")
        assert len(stored) == 2

        target = stored[0]
        results = engine.recall(embedding_text(target), project_id="p1", k=5)

        assert results[0]["id"] == target.id
        assert results[0]["title"] == target.title
        assert results[0]["score"] > 0.99


def test_remember_dedups_on_same_key() -> None:
    """Remembering the same content reuses ids instead of duplicating."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _make_engine(Path(tmp))
        first = engine.remember("session content", project_id="p1")
        second = engine.remember("session content", project_id="p1")

        assert {m.id for m in first} == {m.id for m in second}
        stats = engine.stats("p1")
        assert stats["total"] == 2
        assert stats["by_type"] == {"bug_fix": 1, "convention": 1}


def test_recall_scoped_by_project() -> None:
    """Recall only returns memories from the requested project."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _make_engine(Path(tmp))
        engine.remember("session content", project_id="p1")
        results = engine.recall("anything", project_id="p2", k=5)
        assert results == []

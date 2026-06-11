"""Tests for entity-hint resolution and memory<->entity linking."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from engram.code.entities import CodeEntity
from engram.engine import MemoryEngine
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def _extraction(title: str, entities: list[str]) -> str:
    """Build a bug_fix extraction payload carrying entity hints."""
    return json.dumps(
        [{"type": "bug_fix", "title": title, "body": "b", "details": {}, "entities": entities}]
    )


class FakeClient:
    """Fake LLM: queued extractions, no-op supersession, constant embeddings."""

    def __init__(self, extractions: list[str]) -> None:
        """Queue one extraction reply per remember() call."""
        self._extractions = extractions
        self._ix = 0

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Route by keyword; pop an extraction otherwise."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.6, "rationale": "x"})
        if "relation" in text and "supersede" in text:
            return "[]"
        out = self._extractions[self._ix]
        self._ix += 1
        return out

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return a constant vector for every input."""
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _engine(tmp_path: Path, client: FakeClient) -> MemoryEngine:
    """Build an engine on temp stores."""
    typed: Any = client
    return MemoryEngine(
        client=typed,
        vector_store=ChromaVectorStore(path=str(tmp_path / "chroma")),
        metadata_store=SqliteMetadataStore(path=str(tmp_path / "db.sqlite")),
        settings=None,
    )


def _seed_entities(engine: MemoryEngine, project_id: str = "p1") -> None:
    """Seed an auth.py module entity and its authenticate function."""
    for key, qual, kind in (
        ("auth.py", "", "module"),
        ("auth.py::authenticate", "authenticate", "function"),
    ):
        engine._meta.upsert_entity(
            CodeEntity(
                entity_key=key,
                project_id=project_id,
                path="auth.py",
                qualname=qual,
                kind=kind,
                source_hash="h",
            )
        )


def test_hint_resolves_and_links_by_symbol() -> None:
    """A symbol hint links the memory to the matching function entity."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), FakeClient([_extraction("Token crash", ["authenticate"])]))
        _seed_entities(engine)
        mem = engine.remember("session", project_id="p1")[0]

        assert "auth.py::authenticate" in engine._meta.entities_for_memory(mem.id)
        assert mem.id in engine._meta.memories_for_entity("auth.py::authenticate")


def test_hint_resolves_by_path_basename() -> None:
    """A file-path hint links the memory to the module entity."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), FakeClient([_extraction("Token crash", ["auth.py"])]))
        _seed_entities(engine)
        mem = engine.remember("session", project_id="p1")[0]

        assert "auth.py" in engine._meta.entities_for_memory(mem.id)


def test_unscanned_project_stashes_hints() -> None:
    """With no entities scanned, hints are stashed in details for later backfill."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), FakeClient([_extraction("Token crash", ["authenticate"])]))
        mem = engine.remember("session", project_id="p1")[0]

        assert engine._meta.entities_for_memory(mem.id) == []
        assert mem.details.get("entity_hints") == ["authenticate"]


def test_recall_entity_boost_lifts_linked() -> None:
    """recall(entity=...) boosts the linked memory above an unlinked one."""
    extractions = [
        _extraction("Token decode crash", ["authenticate"]),
        _extraction("Log line format", []),
    ]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), FakeClient(extractions))
        _seed_entities(engine)
        linked = engine.remember("auth session", project_id="p1")[0]
        engine.remember("logging session", project_id="p1")

        results = engine.recall("anything", project_id="p1", k=5, entity="auth.py::authenticate")
        assert results[0]["id"] == linked.id
        assert results[0]["linked"] is True

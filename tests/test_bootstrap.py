"""Tests for project bootstrap (LLM + git mocked; no network/real git)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from engram.engine import MemoryEngine
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def _extraction(entities: list[str]) -> str:
    """An architecture extraction payload with the given entity hints."""
    return json.dumps(
        [
            {
                "type": "architecture",
                "title": "Layered handler design",
                "body": "a.py hosts the request handler layer.",
                "details": {},
                "entities": entities,
            }
        ]
    )


class FakeClient:
    """Routes chat by keyword; constant embeddings."""

    def __init__(self, entities: list[str]) -> None:
        """Store the entity hints the mocked extraction should emit."""
        self._extraction = _extraction(entities)

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return a scripted reply chosen by prompt keywords."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.7, "rationale": "x"})
        if "CURRENT SOURCE" in text:
            return json.dumps({"relation": "outdated", "confidence": 0.9, "rationale": "x"})
        if "relation" in text and "supersede" in text:
            return "[]"
        return self._extraction

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return a constant vector per input."""
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _engine(store_dir: Path, client: FakeClient) -> MemoryEngine:
    """Build an engine with stores under `store_dir`."""
    typed: Any = client
    return MemoryEngine(
        client=typed,
        vector_store=ChromaVectorStore(path=str(store_dir / "chroma")),
        metadata_store=SqliteMetadataStore(path=str(store_dir / "db.sqlite")),
        settings=None,
    )


def test_bootstrap_creates_entities_memories_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap scans entities, extracts doc memories, and links them."""
    monkeypatch.setattr("engram.code.bootstrap.recent_commits", lambda *a, **k: [])
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        proj = root / "proj"
        proj.mkdir()
        store = root / "store"
        store.mkdir()
        (proj / "a.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
        (proj / "README.md").write_text(
            "# Proj\nLayered design; a.py hosts the handler.\n", encoding="utf-8"
        )

        engine = _engine(store, FakeClient(entities=["a.py"]))
        summary = engine.bootstrap(str(proj), project_id="p1")

        assert summary["entities"] >= 2  # module + handler
        assert summary["memories_by_type"].get("architecture", 0) >= 1
        assert summary["links"] >= 1
        assert engine._meta.memories_for_entity("a.py")

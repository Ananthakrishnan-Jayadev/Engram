"""Tests for code-aware sync: surgical recheck, removal, and no-op (LLM/git mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from engram.engine import MemoryEngine
from engram.memory.models import Memory
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore

Spec = tuple[str, str, str, list[str]]  # (type, title, body, entity_hints)


def _extraction(specs: list[Spec]) -> str:
    """Build a multi-memory extraction payload from specs."""
    return json.dumps(
        [
            {"type": t, "title": title, "body": body, "details": {}, "entities": ents}
            for (t, title, body, ents) in specs
        ]
    )


class FakeClient:
    """Routes chat by keyword; rechecks flag (needs_update) unless it's a
    convention, which is reported still_valid."""

    def __init__(self, extraction: str) -> None:
        """Store the canned extraction payload."""
        self._extraction = extraction

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return a scripted reply chosen by prompt keywords."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.6, "rationale": "x"})
        if "CURRENT SOURCE" in text:
            relation = "still_valid" if "convention" in text else "needs_update"
            return json.dumps({"relation": relation, "confidence": 0.9, "rationale": "x"})
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


def _by_title(engine: MemoryEngine, project_id: str, title: str) -> Memory | None:
    """Find a stored memory by title."""
    for memory in engine._meta.all_memories(project_id):
        if memory.title == title:
            return memory
    return None


def _dirs(tmp: str) -> tuple[Path, Path]:
    """Return (project_dir, store_dir) under `tmp`."""
    root = Path(tmp)
    proj = root / "proj"
    proj.mkdir()
    store = root / "store"
    store.mkdir()
    return proj, store


def test_sync_surgical_recheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing one function body reaches its MODULE-linked memories via the
    containment walk and flags them DETERMINISTICALLY (independent of the LLM
    verdict). Both stay active; nothing is retired; recheck is NOT zero."""
    monkeypatch.setattr("engram.code.bootstrap.recent_commits", lambda *a, **k: [])
    specs: list[Spec] = [
        ("architecture", "Discount module", "computes the percentage discount.", ["a.py"]),
        ("convention", "Round prices to 2dp", "a.py rounds prices to two decimals.", ["a.py"]),
    ]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj, store = _dirs(tmp)
        (proj / "a.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
        (proj / "README.md").write_text(
            "# P\nDiscount module in a.py; prices rounded to 2dp.\n", encoding="utf-8"
        )
        engine = _engine(store, FakeClient(_extraction(specs)))
        engine.bootstrap(str(proj), project_id="p1")

        arch = _by_title(engine, "p1", "Discount module")
        conv = _by_title(engine, "p1", "Round prices to 2dp")
        assert arch is not None and conv is not None
        assert "a.py" in engine._meta.entities_for_memory(arch.id)
        assert "a.py" in engine._meta.entities_for_memory(conv.id)

        # Surgical edit: change only the function body (module hash is structural,
        # so only a.py::handler is "changed").
        (proj / "a.py").write_text("def handler():\n    return 999\n", encoding="utf-8")
        report = engine.sync_code(str(proj), project_id="p1")

        assert report["changed"] == 1  # a.py::handler only
        assert report["rechecked"] >= 1  # reconnect: the function reaches the module memories
        assert report["flagged"] >= 1
        assert report["superseded"] == 0

        arch_after = engine._meta.get_memory(arch.id)
        conv_after = engine._meta.get_memory(conv.id)
        # The module-level memory about the function is flagged (still active).
        assert arch_after is not None and arch_after.status == "active"
        assert arch_after.details.get("needs_update") is True
        # The convention's recheck verdict is still_valid, yet it is flagged too:
        # the flag is deterministic, not verdict-dependent. It stays active.
        assert conv_after is not None and conv_after.status == "active"
        assert conv_after.details.get("needs_update") is True


def test_sync_removed_entity_supersedes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a function retires the memory that was about it (gone is gone)."""
    monkeypatch.setattr("engram.code.bootstrap.recent_commits", lambda *a, **k: [])
    specs: list[Spec] = [("bug_fix", "Helper computes tax", "helper() computes tax.", ["helper"])]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj, store = _dirs(tmp)
        (proj / "a.py").write_text(
            "def handler():\n    return 1\n\n\ndef helper():\n    return 2\n", encoding="utf-8"
        )
        (proj / "README.md").write_text("# P\nhelper computes tax in a.py.\n", encoding="utf-8")
        engine = _engine(store, FakeClient(_extraction(specs)))
        engine.bootstrap(str(proj), project_id="p1")

        linked = engine._meta.memories_for_entity("a.py::helper")
        assert linked

        (proj / "a.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
        report = engine.sync_code(str(proj), project_id="p1")

        assert report["removed"] == 1
        assert report["superseded"] == 1
        memory = engine._meta.get_memory(linked[0])
        assert memory is not None and memory.status == "superseded"


def test_sync_unchanged_rechecks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sync with no source changes rechecks and retires nothing."""
    monkeypatch.setattr("engram.code.bootstrap.recent_commits", lambda *a, **k: [])
    specs: list[Spec] = [("bug_fix", "Handler result", "handler() returns it.", ["handler"])]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj, store = _dirs(tmp)
        (proj / "a.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
        (proj / "README.md").write_text("# P\nhandler in a.py.\n", encoding="utf-8")
        engine = _engine(store, FakeClient(_extraction(specs)))
        engine.bootstrap(str(proj), project_id="p1")

        report = engine.sync_code(str(proj), project_id="p1")
        assert report["changed"] == 0
        assert report["rechecked"] == 0
        assert report["superseded"] == 0

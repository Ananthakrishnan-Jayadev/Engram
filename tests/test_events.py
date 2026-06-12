"""Engine decision points persist rows in the events table (LLM mocked)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from engram.engine import MemoryEngine
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = _T0 + timedelta(days=10)


class _Client:
    """Fake LLM: forced supersession relation, scripted everything else."""

    def __init__(self, relation: str = "supersedes") -> None:
        """Configure the relation every supersession verdict carries."""
        self.relation = relation

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Route by prompt keyword."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.6, "rationale": "x"})
        if "CURRENT SOURCE" in text:
            return json.dumps({"relation": "needs_update", "confidence": 0.9, "rationale": "x"})
        if "CANDIDATES" in text:
            ids = re.findall(r"id:\s*(\S+)", text)
            return json.dumps(
                [{"target_id": i, "relation": self.relation, "confidence": 0.95} for i in ids]
            )
        return json.dumps(
            [
                {
                    "type": "bug_fix",
                    "title": "Seeded fix",
                    "body": "b",
                    "details": {},
                    "entities": ["handler"],
                }
            ]
        )

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Constant vector so prior memories are always candidates."""
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _engine(tmp: str, client: _Client) -> MemoryEngine:
    """Engine over temp stores."""
    typed: Any = client
    return MemoryEngine(
        client=typed,
        vector_store=ChromaVectorStore(path=str(Path(tmp) / "chroma")),
        metadata_store=SqliteMetadataStore(path=str(Path(tmp) / "db.sqlite")),
        settings=None,
    )


def _mem(mid: str, created_at: datetime) -> Memory:
    """A bug_fix memory with a controlled creation time."""
    return Memory(
        id=mid,
        project_id="p1",
        type=MemoryType.BUG_FIX,
        title=mid,
        body="b",
        created_at=created_at,
    )


def _kinds(engine: MemoryEngine) -> list[str]:
    """Event kinds for p1 (newest first)."""
    return [e["kind"] for e in engine._meta.list_events("p1", limit=100)]


def test_remember_and_supersede_write_events() -> None:
    """remember + a supersession decision both append events."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(tmp, _Client("supersedes"))
        engine.remember_structured(_mem("old", _T0), "p1")
        engine.remember_structured(_mem("new", _T1), "p1")

        kinds = _kinds(engine)
        assert kinds.count("remember") == 2
        assert "superseded" in kinds

        superseded = next(
            e for e in engine._meta.list_events("p1", limit=100) if e["kind"] == "superseded"
        )
        assert superseded["memory_id"] == "old"
        assert "new" in superseded["detail"]


def test_blocked_supersession_writes_event() -> None:
    """A guard block (candidate not older) records supersession_blocked."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(tmp, _Client("supersedes"))
        engine.remember_structured(_mem("new", _T1), "p1")
        engine.remember_structured(_mem("old", _T0), "p1")  # older arrives second

        kinds = _kinds(engine)
        assert "supersession_blocked" in kinds
        assert "superseded" not in kinds


def test_duplicate_merge_writes_event() -> None:
    """A duplicate verdict records duplicate_merge."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(tmp, _Client("duplicate"))
        engine.remember_structured(_mem("old", _T0), "p1")
        engine.remember_structured(_mem("new", _T1), "p1")
        assert "duplicate_merge" in _kinds(engine)


def test_sync_writes_recheck_and_flag_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """A code edit + sync records recheck and flagged events."""
    monkeypatch.setattr("engram.code.bootstrap.recent_commits", lambda *a, **k: [])
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj = Path(tmp) / "proj"
        proj.mkdir()
        (proj / "a.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
        (proj / "README.md").write_text("# P\nhandler in a.py.\n", encoding="utf-8")

        engine = _engine(tmp, _Client("unrelated"))
        engine.bootstrap(str(proj), project_id="p1")

        (proj / "a.py").write_text("def handler():\n    return 999\n", encoding="utf-8")
        engine.sync_code(str(proj), project_id="p1")

        kinds = _kinds(engine)
        assert "recheck" in kinds
        assert "flagged" in kinds

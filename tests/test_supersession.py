"""Tests for supersession verdicts and engine-level marking (LLM mocked)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from engram.engine import MemoryEngine
from engram.intelligence.supersession import check_supersession
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def _memory(mid: str, title: str) -> Memory:
    """Build a bug_fix memory."""
    return Memory(id=mid, project_id="p1", type=MemoryType.BUG_FIX, title=title, body="b")


def _verdict(target_id: str, relation: str, confidence: float = 0.9) -> str:
    """Build a one-element verdict JSON payload."""
    return json.dumps(
        [{"target_id": target_id, "relation": relation, "confidence": confidence}]
    )


class VerdictClient:
    """Client returning a fixed verdict JSON payload."""

    def __init__(self, payload: str) -> None:
        """Store the canned payload."""
        self.payload = payload

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return the canned payload."""
        return self.payload


def test_parses_supersedes_verdict() -> None:
    """A well-formed supersedes verdict for a known candidate is returned."""
    candidate = _memory("old", "Old fix")
    client: Any = VerdictClient(_verdict("old", "supersedes"))
    verdicts = check_supersession(client, _memory("new", "New fix"), [candidate])
    assert len(verdicts) == 1
    assert verdicts[0].relation == "supersedes"
    assert verdicts[0].target_id == "old"
    assert 0.0 <= verdicts[0].confidence <= 1.0


def test_unknown_target_is_dropped() -> None:
    """Verdicts pointing at unknown candidate ids are discarded."""
    candidate = _memory("old", "Old fix")
    client: Any = VerdictClient(_verdict("ghost", "supersedes"))
    verdicts = check_supersession(client, _memory("new", "New"), [candidate])
    assert verdicts == []


def test_no_candidates_returns_empty() -> None:
    """With no candidates, no LLM call is needed and the result is empty."""
    client: Any = VerdictClient("[]")
    assert check_supersession(client, _memory("new", "New"), []) == []


def _extraction(title: str, body: str) -> str:
    """Build a single-bug_fix extraction JSON payload."""
    return json.dumps([{"type": "bug_fix", "title": title, "body": body, "details": {}}])


class _ScriptedClient:
    """Routes chat by prompt keyword; echoes candidate ids in supersedes verdicts."""

    def __init__(self, extractions: list[str]) -> None:
        """Queue extraction responses (one per remember)."""
        self._extractions = extractions
        self._ix = 0

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return a scripted reply based on the prompt."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.7, "rationale": "scripted"})
        if "relation" in text and "supersede" in text:
            ids = re.findall(r"id:\s*(\S+)", text)
            return json.dumps([_verdict_obj(i) for i in ids])
        out = self._extractions[self._ix]
        self._ix += 1
        return out

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return a constant vector so the prior memory is always a candidate."""
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _verdict_obj(target_id: str) -> dict[str, Any]:
    """Build a supersedes verdict object for `target_id`."""
    return {"target_id": target_id, "relation": "supersedes", "confidence": 0.95}


def test_engine_marks_target_superseded_and_adds_edge() -> None:
    """Remembering a superseding fix marks the old one and records an edge."""
    extractions = [
        _extraction("Handle expired tokens", "old approach"),
        _extraction("Centralized token validation", "new approach"),
    ]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        client: Any = _ScriptedClient(extractions)
        engine = MemoryEngine(
            client=client,
            vector_store=ChromaVectorStore(path=str(tmp_path / "chroma")),
            metadata_store=SqliteMetadataStore(path=str(tmp_path / "engram.sqlite")),
            settings=None,
        )
        old_id = engine.remember("original", project_id="p1")[0].id
        new_id = engine.remember("refactor", project_id="p1")[0].id

        old = engine._meta.get_memory(old_id)
        assert old is not None and old.status == "superseded"
        assert (old_id, "supersedes") in engine._meta.outgoing_edges(new_id)

"""End-to-end intelligence-engine tests: supersession, recall, answer, feedback."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from engram.engine import MemoryEngine
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def _extraction(title: str, body: str, mtype: str = "bug_fix") -> str:
    """Build a single-memory extraction JSON payload of type `mtype`."""
    return json.dumps([{"type": mtype, "title": title, "body": body, "details": {}}])


def _multi_extraction(items: list[tuple[str, str, str]]) -> str:
    """Build a multi-memory extraction payload from (type, title, body) tuples."""
    return json.dumps(
        [{"type": t, "title": ttl, "body": b, "details": {}} for t, ttl, b in items]
    )


class ScriptedClient:
    """A configurable fake LLM client that routes chat calls by prompt keyword."""

    def __init__(
        self,
        extractions: list[str],
        relation: str = "unrelated",
        confidence: float = 0.95,
        salience: float = 0.6,
        fixed_vector: list[float] | None = None,
        salience_by_type: dict[str, float] | None = None,
    ) -> None:
        """Configure scripted extraction, supersession, and embedding behaviour."""
        self._extractions = extractions
        self._ix = 0
        self.relation = relation
        self.confidence = confidence
        self.salience = salience
        self.fixed_vector = fixed_vector
        self.salience_by_type = salience_by_type

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return a scripted reply chosen by keywords in the prompt."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            score = self.salience
            if self.salience_by_type:
                match = re.search(r"type:\s*(\w+)", text)
                if match and match.group(1) in self.salience_by_type:
                    score = self.salience_by_type[match.group(1)]
            return json.dumps({"score": score, "rationale": "scripted"})
        if "relation" in text and "supersede" in text:
            ids = re.findall(r"id:\s*(\S+)", text)
            return json.dumps([self._verdict(i) for i in ids])
        if "synthesize" in text:
            return "Validate the token before decoding; raise Unauthorized if missing."
        if "Condense" in text:
            return "condensed"
        out = self._extractions[self._ix]
        self._ix += 1
        return out

    def _verdict(self, target_id: str) -> dict[str, Any]:
        """Build a verdict object for `target_id`."""
        return {
            "target_id": target_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "rationale": "scripted",
        }

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return deterministic vectors (or a fixed vector if configured)."""
        if self.fixed_vector is not None:
            return [list(self.fixed_vector) for _ in texts]
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Near-orthogonal deterministic vector for `text`."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b - 127.5) / 127.5 for b in digest[:32]]


def _engine(tmp_path: Path, client: ScriptedClient) -> MemoryEngine:
    """Build an engine backed by temp stores and `client`."""
    typed: Any = client
    return MemoryEngine(
        client=typed,
        vector_store=ChromaVectorStore(path=str(tmp_path / "chroma")),
        metadata_store=SqliteMetadataStore(path=str(tmp_path / "engram.sqlite")),
        settings=None,
    )


def test_architecture_supersedes_bugfix() -> None:
    """A superseding architecture memory supersedes the bug_fix and survives recall."""
    extractions = [
        _extraction("Try/except token decode", "wrap decode in try/except"),
        _extraction("Central token validation", "validate before decode", mtype="architecture"),
    ]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), ScriptedClient(extractions, relation="supersedes"))
        bug = engine.remember("original fix", project_id="p1")[0]
        arch = engine.remember("refactor", project_id="p1")[0]

        bug_rec = engine._meta.get_memory(bug.id)
        arch_rec = engine._meta.get_memory(arch.id)
        assert bug_rec is not None and bug_rec.status == "superseded"
        assert arch_rec is not None and arch_rec.status == "active"

        results = engine.recall("how to handle expired tokens", project_id="p1", k=5)
        assert results[0]["id"] == arch.id
        assert results[0]["type"] == "architecture"


def test_rejected_sibling_does_not_supersede_architecture() -> None:
    """A rejected_approach sibling can't retire the architecture that supersedes a bug_fix."""
    batch = _multi_extraction(
        [
            ("architecture", "Central token guard", "validate up front"),
            ("rejected_approach", "Try/except decode hack", "rejected hack"),
        ]
    )
    extractions = [_extraction("Old token decode fix", "patch decode"), batch]
    client = ScriptedClient(
        extractions,
        relation="supersedes",
        fixed_vector=[0.1, 0.2, 0.3, 0.4],
        salience_by_type={"architecture": 0.9, "rejected_approach": 0.3, "bug_fix": 0.6},
    )
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), client)
        bug = engine.remember("legacy session", project_id="p1")[0]
        stored = engine.remember("refactor session", project_id="p1")
        arch = next(m for m in stored if m.type is MemoryType.ARCHITECTURE)
        rejected = next(m for m in stored if m.type is MemoryType.REJECTED_APPROACH)

        bug_rec = engine._meta.get_memory(bug.id)
        arch_rec = engine._meta.get_memory(arch.id)

        assert bug_rec is not None and bug_rec.status == "superseded"  # (a)
        assert arch_rec is not None and arch_rec.status == "active"  # (b)
        # (c) the rejected_approach did not supersede the architecture.
        assert (arch.id, "supersedes") not in engine._meta.outgoing_edges(rejected.id)

        results = engine.recall("token handling", project_id="p1", k=5)
        assert results[0]["id"] == arch.id  # (d)
        assert results[0]["type"] == "architecture"


def test_answer_grounding_excludes_rejected() -> None:
    """answer() grounds only on allowed types; a rejected approach becomes an Avoid note."""
    extractions = [
        _extraction("Validate token in middleware", "guard before decode"),
        _extraction("Hand-roll base64 parsing", "manual parsing", mtype="rejected_approach"),
    ]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), ScriptedClient(extractions, relation="unrelated"))
        good = engine.remember("good approach", project_id="p1")[0]
        bad = engine.remember("bad approach", project_id="p1")[0]

        out = engine.answer("how should I handle tokens?", project_id="p1")
        assert good.id in out["used_memory_ids"]
        assert bad.id not in out["used_memory_ids"]
        assert "Avoid:" in out["answer"]


def test_memory_cannot_supersede_twin() -> None:
    """Re-remembering the same content (its own twin) never supersedes itself."""
    extractions = [_extraction("Same fix", "same body"), _extraction("Same fix", "same body")]
    client = ScriptedClient(extractions, relation="supersedes", fixed_vector=[0.1, 0.2, 0.3, 0.4])
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), client)
        first = engine.remember("x", project_id="p1")[0]
        second = engine.remember("x", project_id="p1")[0]
        assert first.id == second.id  # dedup -> same memory (its own twin)

        rec = engine._meta.get_memory(first.id)
        assert rec is not None and rec.status == "active"
        assert engine.stats("p1")["total"] == 1


def test_reset_project_clears_only_target() -> None:
    """reset_project removes the target project's state but leaves others intact."""
    extractions = [_extraction("A fix", "body a"), _extraction("B fix", "body b")]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), ScriptedClient(extractions))
        engine.remember("a", project_id="p1")
        engine.remember("b", project_id="p2")

        engine.reset_project("p1")

        assert engine.stats("p1")["total"] == 0
        assert engine.stats("p2")["total"] == 1
        assert engine.recall("anything", project_id="p1") == []
        assert engine.recall("b fix", project_id="p2")


def test_answer_synthesizes_from_recall() -> None:
    """answer() recalls, packs, and returns a synthesized answer + used ids."""
    extractions = [_extraction("Expired token handling", "validate before decode")]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), ScriptedClient(extractions))
        engine.remember("token fix session", project_id="p1")

        out = engine.answer("how do I handle expired tokens?", project_id="p1")
        assert out["answer"]
        assert isinstance(out["used_memory_ids"], list)
        assert out["used_memory_ids"]


def test_feedback_shifts_ranking() -> None:
    """Helpful feedback raises a memory's salience and lifts its ranking."""
    extractions = [_extraction("Fix A", "body A"), _extraction("Fix B", "body B")]
    # Identical embeddings make similarity tie, so strength/salience decides order.
    client = ScriptedClient(extractions, relation="unrelated", fixed_vector=[0.1, 0.2, 0.3, 0.4])
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), client)
        engine.remember("first", project_id="p1")
        b_id = engine.remember("second", project_id="p1")[0].id

        for _ in range(4):
            engine.feedback(b_id, helpful=True)

        results = engine.recall("query", project_id="p1", k=5)
        assert results[0]["id"] == b_id


def test_maintenance_marks_stale_dormant() -> None:
    """maintenance transitions a very stale, low-salience memory out of active."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _engine(Path(tmp), ScriptedClient([]))
        old = Memory(
            id="stale",
            project_id="p1",
            type=MemoryType.OPEN_THREAD,
            title="t",
            body="b",
            salience=0.3,
            created_at=datetime.now(UTC) - timedelta(days=120),
        )
        engine._meta.upsert_memory(old)

        report = engine.maintenance("p1")
        assert report["transitions"] >= 1
        refreshed = engine._meta.get_memory("stale")
        assert refreshed is not None and refreshed.status in {"dormant", "forgotten"}

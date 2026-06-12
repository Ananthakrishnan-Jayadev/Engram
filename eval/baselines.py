"""Benchmark strategies: NoMemory, NaiveAll, and the full Engram pipeline.

All strategies consume the same events/queries so the harness can compare them.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from math import sqrt
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from engram.code.entities import scan_project
from engram.engine import MemoryEngine
from engram.intelligence.packing import estimate_tokens
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from eval.generator import CodeEdit, MemorySpec, Scenario

EmbedFn = Callable[[list[str]], list[list[float]]]


@runtime_checkable
class Strategy(Protocol):
    """A memory strategy the harness can replay events into and query."""

    name: str

    def add_memory(self, spec: MemorySpec) -> None:
        """Ingest one structured memory."""
        ...

    def apply_edit(self, edit: CodeEdit) -> None:
        """Apply a code edit (and any consequent maintenance)."""
        ...

    def query(self, text: str, k: int) -> list[str]:
        """Return up to `k` memory keys most relevant to `text`."""
        ...

    def retired_keys(self) -> set[str]:
        """Keys this strategy has retired (superseded)."""
        ...

    def flagged_keys(self) -> set[str]:
        """Keys this strategy has flagged as needing review."""
        ...

    def status_of(self, key: str) -> str:
        """Lifecycle status of `key` (active/dormant/forgotten/superseded/absent)."""
        ...


def _spec_text(spec: MemorySpec) -> str:
    """Canonical text for a spec (matches engine.embedding_text shape)."""
    return f"{spec.type}: {spec.title}\n{spec.body}"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class NoMemory:
    """Baseline with no memory at all — every query returns nothing."""

    name = "no_memory"

    def add_memory(self, spec: MemorySpec) -> None:
        """No-op."""

    def apply_edit(self, edit: CodeEdit) -> None:
        """No-op."""

    def query(self, text: str, k: int) -> list[str]:
        """Always empty."""
        return []

    def retired_keys(self) -> set[str]:
        """Never retires."""
        return set()

    def flagged_keys(self) -> set[str]:
        """Never flags."""
        return set()

    def status_of(self, key: str) -> str:
        """Nothing is stored."""
        return "absent"


class NaiveAll:
    """Stores everything; semantic recall only; never retires or decays."""

    name = "naive_all"

    def __init__(self, embed_fn: EmbedFn) -> None:
        """Use `embed_fn` for semantic ranking."""
        self._embed = embed_fn
        self._keys: list[str] = []
        self._vectors: list[list[float]] = []

    def add_memory(self, spec: MemorySpec) -> None:
        """Store the spec and its embedding."""
        self._keys.append(spec.key)
        self._vectors.append(self._embed([_spec_text(spec)])[0])

    def apply_edit(self, edit: CodeEdit) -> None:
        """Naive memory ignores code edits."""

    def query(self, text: str, k: int) -> list[str]:
        """Rank all stored memories by cosine similarity (stale included)."""
        q = self._embed([text])[0]
        ranked = sorted(
            zip(self._keys, self._vectors, strict=True),
            key=lambda kv: _cosine(q, kv[1]),
            reverse=True,
        )
        return [key for key, _ in ranked[:k]]

    def retired_keys(self) -> set[str]:
        """Never retires."""
        return set()

    def flagged_keys(self) -> set[str]:
        """Never flags."""
        return set()

    def status_of(self, key: str) -> str:
        """Stored memories are always active; never retired or decayed."""
        return "active" if key in self._keys else "absent"


class EngramStrategy:
    """Wraps a MemoryEngine and runs the full extract→store→recall→forget loop."""

    name = "engram"

    def __init__(
        self,
        engine: MemoryEngine,
        project_dir: str | Path,
        project_id: str = "bench",
        mode: str = "direct",
        transcripts: dict[str, str] | None = None,
    ) -> None:
        """Reset `project_id`, scan `project_dir` for entities, and prepare maps.

        mode="direct" injects pre-structured memories (no extraction variance);
        mode="e2e" feeds each memory's raw transcript through Qwen extraction.
        """
        self._engine = engine
        self._dir = Path(project_dir)
        self._pid = project_id
        self._mode = mode
        self._transcripts = transcripts or {}
        self._key_by_id: dict[str, str] = {}
        self._id_by_key: dict[str, str] = {}
        engine.reset_project(project_id)
        for entity in scan_project(self._dir, project_id=project_id):
            engine._meta.upsert_entity(entity)

    def add_memory(self, spec: MemorySpec) -> None:
        """Ingest the spec, either structured (direct) or via extraction (e2e)."""
        if self._mode == "e2e":
            text = self._transcripts.get(spec.key, f"{spec.title}\n{spec.body}")
            stored = self._engine.remember(text, project_id=self._pid, source="bench")
            for memory in stored:
                self._key_by_id[memory.id] = spec.key
            if stored:
                self._id_by_key.setdefault(spec.key, stored[0].id)
            return

        details: dict[str, Any] = dict(spec.details)
        if spec.entity_hints:
            details["entity_hints"] = list(spec.entity_hints)
        memory = Memory(
            id=uuid.uuid4().hex,
            project_id=self._pid,
            type=MemoryType(spec.type),
            title=spec.title,
            body=spec.body,
            details=details,
        )
        stored_memory = self._engine.remember_structured(memory, project_id=self._pid)
        self._key_by_id[stored_memory.id] = spec.key
        self._id_by_key[spec.key] = stored_memory.id

    def apply_edit(self, edit: CodeEdit) -> None:
        """Write the edited source and run a code-aware sync."""
        (self._dir / edit.path).write_text(edit.new_source, encoding="utf-8")
        self._engine.sync_code(str(self._dir), project_id=self._pid)

    def query(self, text: str, k: int) -> list[str]:
        """Recall via the full pipeline and map memory ids back to keys."""
        results = self._engine.recall(text, project_id=self._pid, k=k)
        assert isinstance(results, list)
        return [self._key_by_id.get(r["id"], r["id"]) for r in results]

    def retired_keys(self) -> set[str]:
        """Keys of memories the engine marked superseded."""
        return {
            self._key_by_id[m.id]
            for m in self._engine._meta.all_memories(self._pid)
            if m.status == "superseded" and m.id in self._key_by_id
        }

    def flagged_keys(self) -> set[str]:
        """Keys of memories flagged needs_update by recheck."""
        return {
            self._key_by_id[m.id]
            for m in self._engine._meta.all_memories(self._pid)
            if m.details.get("needs_update") and m.id in self._key_by_id
        }

    def status_of(self, key: str) -> str:
        """Lifecycle status of the memory behind `key` (or 'absent')."""
        memory_id = self._id_by_key.get(key)
        if memory_id is None:
            return "absent"
        memory = self._engine._meta.get_memory(memory_id)
        return memory.status if memory is not None else "absent"

    def packing_stats(self, scenario: Scenario, k: int) -> dict[str, float]:
        """Aggregate packed-vs-naive token cost and gold retention over queries."""
        packed = naive = gold_in = total = 0
        for query in scenario.queries:
            res = self._engine.recall(query.text, project_id=self._pid, k=k, pack=True)
            assert isinstance(res, dict)
            context = res["context"]
            packed += int(context["est_tokens"])
            naive += sum(
                estimate_tokens(f"[{r['type']}] {r['title']}\n{r['body']}") for r in res["results"]
            )
            total += 1
            gold_id = self._id_by_key.get(query.gold_key)
            if gold_id and gold_id in context["included_ids"]:
                gold_in += 1
        from eval.metrics import packing_efficiency

        return packing_efficiency(packed, naive, gold_in, total)

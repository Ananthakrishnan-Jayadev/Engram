"""Re-scan code, detect changed/removed entities, and recheck linked memories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engram.code.entities import ancestor_keys, entity_source, scan_project
from engram.llm.models import heavy_model_for
from engram.llm.parsing import request_json
from engram.prompts import CODE_RECHECK_SYSTEM, build_code_recheck_prompt

if TYPE_CHECKING:
    from engram.engine import MemoryEngine
    from engram.llm.client import QwenClient
    from engram.memory.models import Memory

# Durable memory types resist retirement: they need very high confidence to
# supersede, since an implementation change rarely invalidates them.
DURABLE_TYPES = {"architecture", "convention"}
SUPERSEDE_CONFIDENCE_DURABLE = 0.85
SUPERSEDE_CONFIDENCE_CODE = 0.70
# A "needs_update" (or sub-threshold "outdated") verdict scales salience by this.
NEEDS_UPDATE_SALIENCE_FACTOR = 0.7


def _recheck_one(
    client: QwenClient, memory: Memory, entity_key: str, source: str
) -> dict[str, Any]:
    """Ask the heavy model whether `memory` still holds for `source`."""
    messages = [
        {"role": "system", "content": CODE_RECHECK_SYSTEM},
        {
            "role": "user",
            "content": build_code_recheck_prompt(
                memory.type.value, memory.title, memory.body, entity_key, source
            ),
        },
    ]
    data = request_json(client, messages, model=heavy_model_for(client))
    return data if isinstance(data, dict) else {}


def _flag_needs_update(engine: MemoryEngine, memory: Memory) -> None:
    """Keep `memory` active but lower its salience and flag it for update."""
    memory.salience = max(0.0, memory.salience * NEEDS_UPDATE_SALIENCE_FACTOR)
    memory.details = {**memory.details, "needs_update": True}
    engine._meta.upsert_memory(memory)


def _apply_verdict(engine: MemoryEngine, memory: Memory, verdict: dict[str, Any]) -> str:
    """Apply a recheck verdict to `memory`, returning the action taken.

    Supersession stays LLM-gated and conservative (durable types need very high
    confidence). The needs_update flag, however, is DETERMINISTIC: any memory
    rechecked because its linked code changed is flagged for review regardless of
    the (unreliable) verdict.
    """
    relation = verdict.get("relation")
    try:
        confidence = float(verdict.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    threshold = (
        SUPERSEDE_CONFIDENCE_DURABLE
        if memory.type.value in DURABLE_TYPES
        else SUPERSEDE_CONFIDENCE_CODE
    )
    if relation == "outdated" and confidence >= threshold:
        engine._meta.set_status(memory.id, "superseded")
        return "superseded"
    # Deterministic: the referenced code changed, so flag for review.
    _flag_needs_update(engine, memory)
    return "flagged"


def _changed_target_ids(engine: MemoryEngine, entity_key: str) -> list[str]:
    """Memory ids to recheck for a changed entity: itself plus its ancestors.

    A changed function is by definition a change to its class/module contents,
    so memories linked one level up (the common case) are reached too.
    """
    ids: list[str] = []
    for key in (entity_key, *ancestor_keys(entity_key)):
        ids.extend(engine._meta.memories_for_entity(key))
    return ids


def _supersede_removed(engine: MemoryEngine, entity_key: str, counters: dict[str, int]) -> None:
    """A removed entity retires its linked memories outright (gone is gone)."""
    memory_ids = engine._meta.memories_for_entity(entity_key)
    for memory in engine._meta.get_memories(memory_ids, include_inactive=True):
        if memory.status == "superseded":
            continue
        engine._meta.set_status(memory.id, "superseded")
        counters["superseded"] += 1
        engine._record_event(
            memory.project_id,
            "superseded",
            memory.id,
            f"linked entity {entity_key} was removed",
        )


def _clear_stale_flags(engine: MemoryEngine, project_id: str, flagged_ids: set[str]) -> None:
    """Clear needs_update on active memories not (re)flagged this run.

    A memory only stays flagged while its linked code keeps changing; if this
    sync did not flag it, its linked entities were unchanged, so clear it.
    """
    for memory in engine._meta.all_memories(project_id):
        if memory.status != "active" or memory.id in flagged_ids:
            continue
        if memory.details.get("needs_update"):
            memory.details = {k: v for k, v in memory.details.items() if k != "needs_update"}
            engine._meta.upsert_memory(memory)


def run_sync(engine: MemoryEngine, project_id: str, project_path: str) -> dict[str, Any]:
    """Re-scan code, diff hashes, recheck affected memories, update the graph.

    Returns {changed, removed, new, rechecked, superseded, flagged}.
    """
    stored = {e.entity_key: e for e in engine._meta.list_entities(project_id)}
    scanned = scan_project(project_path, project_id=project_id)
    scanned_by_key = {e.entity_key: e for e in scanned}

    changed = [
        key
        for key, entity in scanned_by_key.items()
        if key in stored and stored[key].source_hash != entity.source_hash
    ]
    new = [key for key in scanned_by_key if key not in stored]
    removed = [key for key in stored if key not in scanned_by_key]

    counters = {"rechecked": 0, "superseded": 0, "flagged": 0}
    rechecked_ids: set[str] = set()
    flagged_ids: set[str] = set()
    for key in changed:
        source = entity_source(project_path, scanned_by_key[key]) or ""
        memory_ids = _changed_target_ids(engine, key)
        if not memory_ids:
            continue
        for memory in engine._meta.get_memories(memory_ids, include_inactive=True):
            if memory.id in rechecked_ids or memory.status == "superseded":
                continue
            rechecked_ids.add(memory.id)
            verdict = _recheck_one(engine._client, memory, key, source)
            counters["rechecked"] += 1
            engine._record_event(project_id, "recheck", memory.id, f"entity {key} changed")
            action = _apply_verdict(engine, memory, verdict)
            if action == "superseded":
                counters["superseded"] += 1
                engine._record_event(
                    project_id,
                    "superseded",
                    memory.id,
                    f"recheck: outdated vs current source of {key}",
                )
            else:  # deterministic flag
                counters["flagged"] += 1
                flagged_ids.add(memory.id)
                engine._record_event(
                    project_id,
                    "flagged",
                    memory.id,
                    f"needs_update: {key} changed",
                )
    for key in removed:
        _supersede_removed(engine, key, counters)

    _clear_stale_flags(engine, project_id, flagged_ids)

    # Update stored hashes: upsert all current entities; drop removed ones.
    for entity in scanned:
        engine._meta.upsert_entity(entity)
    for key in removed:
        engine._meta.delete_entity(key)

    # New entities may resolve previously-stashed entity hints.
    engine.backfill_entity_links(project_id)

    return {
        "changed": len(changed),
        "removed": len(removed),
        "new": len(new),
        **counters,
    }

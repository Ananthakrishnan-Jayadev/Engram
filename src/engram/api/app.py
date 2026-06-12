"""FastAPI read API over the SQLite store Engram writes.

Read-only: it opens the same `engram.sqlite` the engine uses and serves
memories, the knowledge graph, the event log, stats, decay curves, and the
latest benchmark metrics for the dashboard.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from engram.intelligence.decay import effective_strength
from engram.memory.models import Memory
from engram.storage.metadata_store import SqliteMetadataStore

# Decay-curve sampling: points per memory, days beyond "now" to project.
DECAY_SAMPLES = 16
DECAY_HORIZON_DAYS = 30.0


def _memory_payload(store: SqliteMetadataStore, memory: Memory) -> dict[str, Any]:
    """Serialise a memory (plus its linked entity keys) for the API."""
    return {
        "id": memory.id,
        "type": memory.type.value,
        "title": memory.title,
        "body": memory.body,
        "status": memory.status,
        "salience": memory.salience,
        "created_at": memory.created_at.isoformat(),
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
        "details": memory.details,
        "needs_update": bool(memory.details.get("needs_update")),
        "entities": store.entities_for_memory(memory.id),
    }


def _decay_points(memory: Memory, now: datetime) -> list[dict[str, Any]]:
    """Sample effective_strength from creation to now+horizon."""
    start = memory.created_at
    end = now + timedelta(days=DECAY_HORIZON_DAYS)
    if end <= start:
        end = start + timedelta(days=1)
    step = (end - start) / (DECAY_SAMPLES - 1)
    return [
        {
            "t": (start + step * i).isoformat(),
            "strength": round(effective_strength(memory, start + step * i), 4),
        }
        for i in range(DECAY_SAMPLES)
    ]


def create_app(
    sqlite_path: str | None = None,
    metrics_path: str | Path = "eval/results/latest.json",
) -> FastAPI:
    """Build the read API. Paths default to settings / repo conventions."""
    app = FastAPI(title="Engram API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    state: dict[str, SqliteMetadataStore | None] = {"store": None}

    def store() -> SqliteMetadataStore:
        """Lazily open the metadata store (settings resolved on first use)."""
        if state["store"] is None:
            path = sqlite_path
            if path is None:
                from engram.config import get_settings

                path = get_settings().sqlite_path
            opened = SqliteMetadataStore(path)
            opened.init()
            state["store"] = opened
        return state["store"]

    def _get_memory_or_404(memory_id: str) -> Memory:
        memory = store().get_memory(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return memory

    @app.get("/api/projects")
    def projects() -> list[str]:
        """Distinct project ids."""
        return store().list_projects()

    @app.get("/api/projects/{pid}/memories")
    def memories(pid: str, type: str | None = None, status: str | None = None) -> list[dict]:
        """All memories for a project, optionally filtered by type/status."""
        records = store().all_memories(pid)
        if type:
            records = [m for m in records if m.type.value == type]
        if status:
            records = [m for m in records if m.status == status]
        records.sort(key=lambda m: m.created_at, reverse=True)
        return [_memory_payload(store(), m) for m in records]

    @app.get("/api/projects/{pid}/memories/{memory_id}")
    def memory_detail(pid: str, memory_id: str) -> dict:
        """One memory plus its explainability (edges, entities, flags)."""
        memory = _get_memory_or_404(memory_id)
        s = store()
        return {
            **_memory_payload(s, memory),
            "supersedes": [{"id": dst, "kind": kind} for dst, kind in s.outgoing_edges(memory_id)],
            "superseded_by": [
                {"id": src, "kind": kind} for src, kind in s.incoming_edges(memory_id)
            ],
        }

    @app.get("/api/projects/{pid}/graph")
    def graph(pid: str) -> dict:
        """React-Flow-shaped graph: memory + entity nodes, supersedes + reference edges."""
        s = store()
        mems = s.all_memories(pid)
        entities = s.list_entities(pid)

        nodes: list[dict[str, Any]] = []
        for i, entity in enumerate(entities):
            nodes.append(
                {
                    "id": entity.entity_key,
                    "type": "entity",
                    "position": {"x": 0, "y": i * 90},
                    "data": {
                        "label": entity.entity_key,
                        "kind": entity.kind,
                        "path": entity.path,
                    },
                }
            )
        for i, memory in enumerate(mems):
            nodes.append(
                {
                    "id": memory.id,
                    "type": "memory",
                    "position": {"x": 420 + (i % 3) * 320, "y": (i // 3) * 140},
                    "data": {
                        "label": memory.title,
                        "memoryType": memory.type.value,
                        "status": memory.status,
                        "salience": memory.salience,
                        "needsUpdate": bool(memory.details.get("needs_update")),
                    },
                }
            )

        edges: list[dict[str, Any]] = []
        for src, dst, kind in s.all_edges([m.id for m in mems]):
            edges.append(
                {
                    "id": f"{src}->{dst}:{kind}",
                    "source": src,
                    "target": dst,
                    "kind": kind,
                    "animated": kind == "supersedes",
                }
            )
        for memory in mems:
            for entity_key in s.entities_for_memory(memory.id):
                edges.append(
                    {
                        "id": f"{memory.id}->{entity_key}:ref",
                        "source": memory.id,
                        "target": entity_key,
                        "kind": "references",
                        "animated": False,
                    }
                )
        return {"nodes": nodes, "edges": edges}

    @app.get("/api/projects/{pid}/events")
    def events(pid: str, limit: int = 100) -> list[dict]:
        """The decision-event log, newest first."""
        return store().list_events(pid, limit=limit)

    @app.get("/api/projects/{pid}/stats")
    def stats(pid: str) -> dict:
        """Memory counts by type and by status."""
        records = store().all_memories(pid)
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for memory in records:
            by_type[memory.type.value] = by_type.get(memory.type.value, 0) + 1
            by_status[memory.status] = by_status.get(memory.status, 0) + 1
        return {
            "project_id": pid,
            "total": len(records),
            "by_type": by_type,
            "by_status": by_status,
        }

    @app.get("/api/metrics")
    def metrics() -> dict:
        """Latest benchmark results, or {} if none have been written yet."""
        path = Path(metrics_path)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @app.get("/api/projects/{pid}/decay")
    def decay(pid: str) -> list[dict]:
        """Sampled effective-strength curves per non-superseded memory."""
        now = datetime.now(UTC)
        return [
            {
                "id": memory.id,
                "title": memory.title,
                "type": memory.type.value,
                "status": memory.status,
                "points": _decay_points(memory, now),
            }
            for memory in store().all_memories(pid)
            if memory.status != "superseded"
        ]

    return app


app = create_app()


def main() -> None:
    """Run the API server (the human runs this; not invoked by tests)."""
    import uvicorn

    uvicorn.run("engram.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

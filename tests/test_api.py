"""FastAPI read-API tests against a temp seeded SQLite (offline)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from engram.api.app import create_app
from engram.code.entities import CodeEntity
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore

NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _seed(path: str) -> SqliteMetadataStore:
    """Seed a store with two memories, an edge, an entity+link, and events."""
    store = SqliteMetadataStore(path)
    store.init()
    store.upsert_memory(
        Memory(
            id="m-old",
            project_id="p1",
            type=MemoryType.BUG_FIX,
            title="Old fix",
            body="int cast",
            status="superseded",
            created_at=NOW - timedelta(days=10),
            salience=0.5,
        )
    )
    store.upsert_memory(
        Memory(
            id="m-new",
            project_id="p1",
            type=MemoryType.BUG_FIX,
            title="New fix",
            body="round(total, 2)",
            status="active",
            created_at=NOW,
            salience=0.8,
            details={"needs_update": True},
        )
    )
    store.add_edge("m-new", "m-old", "supersedes")
    store.upsert_entity(
        CodeEntity(
            entity_key="cart.py::checkout",
            project_id="p1",
            path="cart.py",
            qualname="checkout",
            kind="function",
            source_hash="h",
        )
    )
    store.link_memory_entity("m-new", "cart.py::checkout", "p1")
    store.record_event("p1", "remember", "m-old", "[bug_fix] Old fix")
    store.record_event("p1", "superseded", "m-old", "by 'New fix'")
    return store


def _client(tmp: str, metrics: dict | None = None) -> TestClient:
    """Build a TestClient over a seeded temp store."""
    db = str(Path(tmp) / "api.sqlite")
    _seed(db)
    metrics_path = Path(tmp) / "latest.json"
    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return TestClient(create_app(sqlite_path=db, metrics_path=metrics_path))


def test_projects_and_stats() -> None:
    """Projects list and per-project counts by type/status."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        assert client.get("/api/projects").json() == ["p1"]

        stats = client.get("/api/projects/p1/stats").json()
        assert stats["total"] == 2
        assert stats["by_type"] == {"bug_fix": 2}
        assert stats["by_status"] == {"active": 1, "superseded": 1}


def test_memories_list_filters_and_shape() -> None:
    """Memories carry the documented fields; type/status filters apply."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        items = client.get("/api/projects/p1/memories").json()
        assert len(items) == 2
        first = items[0]  # newest first
        assert first["id"] == "m-new"
        expected_keys = {
            "id",
            "type",
            "title",
            "body",
            "status",
            "salience",
            "created_at",
            "access_count",
            "last_accessed",
            "details",
            "needs_update",
            "entities",
        }
        assert expected_keys <= set(first)
        assert first["needs_update"] is True
        assert first["entities"] == ["cart.py::checkout"]

        active = client.get("/api/projects/p1/memories", params={"status": "active"}).json()
        assert [m["id"] for m in active] == ["m-new"]
        none = client.get("/api/projects/p1/memories", params={"type": "convention"}).json()
        assert none == []


def test_memory_detail_explainability() -> None:
    """Detail includes supersedes / superseded-by edges; 404 for unknown ids."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        detail = client.get("/api/projects/p1/memories/m-new").json()
        assert detail["supersedes"] == [{"id": "m-old", "kind": "supersedes"}]
        assert detail["superseded_by"] == []

        old = client.get("/api/projects/p1/memories/m-old").json()
        assert old["superseded_by"] == [{"id": "m-new", "kind": "supersedes"}]

        assert client.get("/api/projects/p1/memories/ghost").status_code == 404


def test_graph_shape() -> None:
    """Graph has typed nodes with positions and supersedes/references edges."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        graph = client.get("/api/projects/p1/graph").json()
        kinds = {n["type"] for n in graph["nodes"]}
        assert kinds == {"memory", "entity"}
        assert all("position" in n and "data" in n for n in graph["nodes"])

        edge_kinds = {e["kind"] for e in graph["edges"]}
        assert edge_kinds == {"supersedes", "references"}
        supersede = next(e for e in graph["edges"] if e["kind"] == "supersedes")
        assert supersede["animated"] is True
        assert supersede["source"] == "m-new" and supersede["target"] == "m-old"


def test_events_newest_first() -> None:
    """Events are returned newest first with the documented shape."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        events = client.get("/api/projects/p1/events", params={"limit": 10}).json()
        assert len(events) == 2
        assert events[0]["kind"] == "superseded"  # recorded last -> first
        assert {"id", "ts", "project_id", "kind", "memory_id", "detail"} <= set(events[0])


def test_metrics_endpoint() -> None:
    """Metrics returns the results JSON, or {} when absent."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        empty = _client(tmp)
        assert empty.get("/api/metrics").json() == {}
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        payload = {"seed": 0, "strategies": {"engram": {"stale_hit_rate": 0.0}}}
        client = _client(tmp, metrics=payload)
        assert client.get("/api/metrics").json() == payload


def test_decay_curves() -> None:
    """Decay excludes superseded memories and yields bounded strength points."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = _client(tmp)
        curves = client.get("/api/projects/p1/decay").json()
        assert [c["id"] for c in curves] == ["m-new"]
        points = curves[0]["points"]
        assert len(points) >= 2
        assert all(0.0 <= p["strength"] <= 1.0 for p in points)
        # Strength decays monotonically (no accesses in the seed data).
        assert points[-1]["strength"] <= points[0]["strength"]

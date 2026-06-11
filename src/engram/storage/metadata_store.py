"""SQLite-backed metadata and knowledge-graph store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from engram.config import get_settings
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.base import StorageInterface
from engram.storage.schema import ADDABLE_MEMORY_COLUMNS, SCHEMA_STATEMENTS


class SqliteMetadataStore(StorageInterface):
    """Authoritative store for memory records and knowledge-graph edges."""

    def __init__(self, path: str | None = None) -> None:
        """Open (or create) the SQLite database at `path` (defaults to settings)."""
        self._path = path or get_settings().sqlite_path

    def _connect(self) -> sqlite3.Connection:
        """Return a connection with row access by column name."""
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        """Create tables if they do not exist, then migrate in any new columns."""
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.commit()
        self.migrate()

    def migrate(self) -> None:
        """Add any missing `memories` columns in place (upgrades dev DBs, no wipe)."""
        with self._connect() as conn:
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
            for name, ddl in ADDABLE_MEMORY_COLUMNS:
                if name not in existing:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {ddl}")
            conn.commit()

    def upsert_memory(self, record: Memory) -> None:
        """Insert or replace a memory record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, project_id, type, title, body, created_at,
                     salience, decay_state, status, access_count, last_accessed,
                     source, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_id = excluded.project_id,
                    type = excluded.type,
                    title = excluded.title,
                    body = excluded.body,
                    created_at = excluded.created_at,
                    salience = excluded.salience,
                    decay_state = excluded.decay_state,
                    status = excluded.status,
                    access_count = excluded.access_count,
                    last_accessed = excluded.last_accessed,
                    source = excluded.source,
                    details = excluded.details
                """,
                (
                    record.id,
                    record.project_id,
                    record.type.value,
                    record.title,
                    record.body,
                    record.created_at.isoformat(),
                    record.salience,
                    record.decay_state,
                    record.status,
                    record.access_count,
                    record.last_accessed.isoformat() if record.last_accessed else None,
                    record.source,
                    json.dumps(record.details),
                ),
            )
            conn.commit()

    def get_memory(self, id: str) -> Memory | None:
        """Fetch a memory record by id, or `None` if it does not exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_memories(self, ids: list[str], include_inactive: bool = False) -> list[Memory]:
        """Batch-fetch records for `ids`; non-active are excluded by default."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
        if not include_inactive:
            sql += " AND status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(sql, ids).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def active_memories(self, project_id: str) -> list[Memory]:
        """Return all active memories for `project_id`."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE project_id = ? AND status = 'active'",
                (project_id,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def all_memories(self, project_id: str) -> list[Memory]:
        """Return every memory for `project_id`, regardless of status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE project_id = ?", (project_id,)
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def set_status(self, id: str, status: str) -> None:
        """Set the lifecycle status of a memory."""
        with self._connect() as conn:
            conn.execute("UPDATE memories SET status = ? WHERE id = ?", (status, id))
            conn.commit()

    def update_salience(self, id: str, value: float) -> None:
        """Set the salience of a memory."""
        with self._connect() as conn:
            conn.execute("UPDATE memories SET salience = ? WHERE id = ?", (value, id))
            conn.commit()

    def update_access(self, id: str) -> None:
        """Increment access_count and stamp last_accessed = now (reinforcement)."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                """,
                (now, id),
            )
            conn.commit()

    def log_feedback(self, memory_id: str, helpful: bool) -> None:
        """Record a feedback event in the feedback table."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (id, memory_id, helpful, ts) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, memory_id, int(helpful), datetime.now(UTC).isoformat()),
            )
            conn.commit()

    def outgoing_edges(self, src_id: str) -> list[tuple[str, str]]:
        """Return (dst_id, kind) edges originating from `src_id`."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dst_id, kind FROM edges WHERE src_id = ?", (src_id,)
            ).fetchall()
        return [(row["dst_id"], row["kind"]) for row in rows]

    def count_by_type(self, project_id: str) -> dict[str, int]:
        """Return a mapping of memory type -> count for `project_id`."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT type, COUNT(*) AS n
                FROM memories WHERE project_id = ?
                GROUP BY type
                """,
                (project_id,),
            ).fetchall()
        return {row["type"]: row["n"] for row in rows}

    def find_by_key(self, project_id: str, type: str, title: str) -> Memory | None:
        """Find a memory by its (project_id, type, title) key, for dedup."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM memories
                WHERE project_id = ? AND type = ? AND title = ?
                LIMIT 1
                """,
                (project_id, type, title),
            ).fetchone()
        return self._row_to_memory(row) if row is not None else None

    def reset_project(self, project_id: str) -> None:
        """Delete all memories, edges, and feedback belonging to `project_id`."""
        with self._connect() as conn:
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM memories WHERE project_id = ?", (project_id,)
                )
            ]
            conn.execute("DELETE FROM memories WHERE project_id = ?", (project_id,))
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM edges WHERE src_id IN ({placeholders}) "
                    f"OR dst_id IN ({placeholders})",
                    ids + ids,
                )
                conn.execute(
                    f"DELETE FROM feedback WHERE memory_id IN ({placeholders})", ids
                )
            conn.commit()

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        """Add a knowledge-graph edge (idempotent on the (src, dst, kind) key)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO edges (src_id, dst_id, kind)
                VALUES (?, ?, ?)
                """,
                (src, dst, kind),
            )
            conn.commit()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        """Convert a SQLite row into a `Memory` model."""
        keys = row.keys()
        raw_details = row["details"] if "details" in keys else None
        try:
            details = json.loads(raw_details) if raw_details else {}
        except (json.JSONDecodeError, TypeError):
            details = {}
        last_accessed_raw = row["last_accessed"] if "last_accessed" in keys else None
        return Memory(
            id=row["id"],
            project_id=row["project_id"],
            type=MemoryType(row["type"]),
            title=row["title"],
            body=row["body"],
            created_at=datetime.fromisoformat(row["created_at"]),
            salience=row["salience"],
            decay_state=row["decay_state"],
            status=row["status"] if "status" in keys else "active",
            access_count=row["access_count"] if "access_count" in keys else 0,
            last_accessed=(
                datetime.fromisoformat(last_accessed_raw) if last_accessed_raw else None
            ),
            source=row["source"],
            details=details,
        )

    # --- Vector ops live in ChromaVectorStore -----------------------------
    def add_vector(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Not handled by the metadata store (see ChromaVectorStore)."""
        raise NotImplementedError("Use ChromaVectorStore for vectors.")

    def query(
        self, embedding: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Not handled by the metadata store (see ChromaVectorStore)."""
        raise NotImplementedError("Use ChromaVectorStore for vector queries.")

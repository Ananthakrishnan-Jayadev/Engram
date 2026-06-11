"""SQLite-backed metadata and knowledge-graph store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from engram.config import get_settings
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.base import StorageInterface
from engram.storage.schema import SCHEMA_STATEMENTS


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
        """Create tables if they do not already exist."""
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def upsert_memory(self, record: Memory) -> None:
        """Insert or replace a memory record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, project_id, type, title, body, created_at,
                     salience, decay_state, source, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_id = excluded.project_id,
                    type = excluded.type,
                    title = excluded.title,
                    body = excluded.body,
                    created_at = excluded.created_at,
                    salience = excluded.salience,
                    decay_state = excluded.decay_state,
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

    def get_memories(self, ids: list[str]) -> list[Memory]:
        """Batch-fetch memory records for `ids` (missing ids are skipped)."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

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
        raw_details = row["details"] if "details" in row.keys() else None
        try:
            details = json.loads(raw_details) if raw_details else {}
        except (json.JSONDecodeError, TypeError):
            details = {}
        return Memory(
            id=row["id"],
            project_id=row["project_id"],
            type=MemoryType(row["type"]),
            title=row["title"],
            body=row["body"],
            created_at=datetime.fromisoformat(row["created_at"]),
            salience=row["salience"],
            decay_state=row["decay_state"],
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

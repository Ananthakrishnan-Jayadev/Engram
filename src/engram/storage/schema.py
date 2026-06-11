"""SQLite DDL for Engram's metadata and knowledge-graph tables."""

from __future__ import annotations

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    type        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    salience    REAL NOT NULL DEFAULT 1.0,
    decay_state TEXT NOT NULL DEFAULT 'active',
    source      TEXT NOT NULL DEFAULT 'unknown',
    details     TEXT NOT NULL DEFAULT '{}'
);
"""

CREATE_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind   TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind)
);
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (CREATE_MEMORIES_TABLE, CREATE_EDGES_TABLE)

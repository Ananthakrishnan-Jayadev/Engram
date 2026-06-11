"""SQLite DDL for Engram's metadata, knowledge-graph, and feedback tables."""

from __future__ import annotations

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    type         TEXT NOT NULL,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    salience     REAL NOT NULL DEFAULT 0.5,
    decay_state  TEXT NOT NULL DEFAULT 'active',
    status       TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    source       TEXT NOT NULL DEFAULT 'unknown',
    details      TEXT NOT NULL DEFAULT '{}'
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

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id        TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    helpful   INTEGER NOT NULL,
    ts        TEXT NOT NULL
);
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    CREATE_MEMORIES_TABLE,
    CREATE_EDGES_TABLE,
    CREATE_FEEDBACK_TABLE,
)

# Columns that migrate() can add in-place to an existing `memories` table.
# Each value is the column DDL appended after `ALTER TABLE ... ADD COLUMN`.
ADDABLE_MEMORY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("salience", "REAL NOT NULL DEFAULT 0.5"),
    ("decay_state", "TEXT NOT NULL DEFAULT 'active'"),
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("access_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_accessed", "TEXT"),
    ("source", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("details", "TEXT NOT NULL DEFAULT '{}'"),
)

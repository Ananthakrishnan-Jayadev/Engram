"""Pydantic data model for a single memory record."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from engram.memory.types import MemoryType


class Memory(BaseModel):
    """A single typed memory about a project."""

    id: str
    project_id: str
    type: MemoryType
    title: str
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    salience: float = 0.5
    decay_state: str = "active"  # legacy; `status` is authoritative
    status: str = "active"  # active | dormant | superseded | forgotten
    access_count: int = 0
    last_accessed: datetime | None = None
    source: str = "unknown"
    details: dict = {}
    embedding: list[float] | None = None

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
    salience: float = 1.0
    decay_state: str = "active"
    source: str = "unknown"
    details: dict = {}
    embedding: list[float] | None = None

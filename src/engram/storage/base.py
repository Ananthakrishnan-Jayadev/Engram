"""Abstract storage contract shared by the vector and metadata layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from engram.memory.models import Memory


class StorageInterface(ABC):
    """Contract for Engram's persistence layers.

    Vector ops are implemented by the Chroma store; metadata/graph ops are
    implemented by the SQLite store. Concrete classes implement the relevant
    half and may leave the others raising :class:`NotImplementedError`.
    """

    @abstractmethod
    def init(self) -> None:
        """Initialise the backing store (create files, tables, collections)."""

    # --- Vector operations -------------------------------------------------
    @abstractmethod
    def add_vector(
        self, id: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Store a precomputed `embedding` under `id` with `metadata`."""

    @abstractmethod
    def query(
        self, embedding: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return the `k` nearest stored vectors, optionally filtered by `where`."""

    # --- Metadata / graph operations --------------------------------------
    @abstractmethod
    def upsert_memory(self, record: Memory) -> None:
        """Insert or update a memory record."""

    @abstractmethod
    def get_memory(self, id: str) -> Memory | None:
        """Fetch a memory record by id, or `None` if absent."""

    @abstractmethod
    def add_edge(self, src: str, dst: str, kind: str) -> None:
        """Add a knowledge-graph edge from `src` to `dst` of type `kind`."""

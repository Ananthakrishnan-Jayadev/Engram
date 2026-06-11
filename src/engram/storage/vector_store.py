"""Chroma-backed vector store using precomputed Qwen embeddings."""

from __future__ import annotations

from typing import Any

import chromadb

from engram.config import get_settings
from engram.memory.models import Memory
from engram.storage.base import StorageInterface

_COLLECTION_NAME = "engram_memories"


class ChromaVectorStore(StorageInterface):
    """Persistent Chroma store for semantic search over precomputed embeddings.

    Embeddings are supplied by Engram (Qwen); Chroma's default embedding
    function is intentionally never used.
    """

    def __init__(self, path: str | None = None) -> None:
        """Create a persistent client rooted at `path` (defaults to settings)."""
        self._path = path or get_settings().chroma_path
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def init(self) -> None:
        """No-op: the collection is created eagerly in `__init__`."""

    def add_vector(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Upsert a precomputed `embedding` under `id` with `metadata`."""
        self._collection.upsert(ids=[id], embeddings=[embedding], metadatas=[metadata])

    def query(
        self, embedding: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Return up to `k` nearest matches as (id, cosine_distance) pairs."""
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            (_id, float(distances[i]) if i < len(distances) else 1.0)
            for i, _id in enumerate(ids)
        ]

    def query_candidates(
        self, embedding: list[float], project_id: str, k: int
    ) -> list[tuple[str, float]]:
        """Return up to `k` neighbours within `project_id` (for supersession)."""
        return self.query(embedding, k, where={"project_id": project_id})

    def reset_project(self, project_id: str) -> None:
        """Delete all vectors belonging to `project_id` (best-effort)."""
        try:
            self._collection.delete(where={"project_id": project_id})
        except Exception:  # noqa: BLE001 - nothing to delete is not an error
            pass

    # --- Metadata/graph ops live in SqliteMetadataStore -------------------
    def upsert_memory(self, record: Memory) -> None:
        """Not handled by the vector store (see SqliteMetadataStore)."""
        raise NotImplementedError("Use SqliteMetadataStore for memory records.")

    def get_memory(self, id: str) -> Memory | None:
        """Not handled by the vector store (see SqliteMetadataStore)."""
        raise NotImplementedError("Use SqliteMetadataStore for memory records.")

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        """Not handled by the vector store (see SqliteMetadataStore)."""
        raise NotImplementedError("Use SqliteMetadataStore for graph edges.")

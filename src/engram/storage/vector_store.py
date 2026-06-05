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
        )

    def init(self) -> None:
        """No-op: the collection is created eagerly in `__init__`."""

    def add_vector(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Upsert a precomputed `embedding` under `id` with `metadata`."""
        self._collection.upsert(ids=[id], embeddings=[embedding], metadatas=[metadata])

    def query(
        self, embedding: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return up to `k` nearest matches, optionally filtered by `where`."""
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for i, _id in enumerate(ids):
            hits.append(
                {
                    "id": _id,
                    "distance": distances[i] if i < len(distances) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else None,
                }
            )
        return hits

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

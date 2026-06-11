"""MemoryEngine: orchestrates the Phase 1 capture path (extract → store → recall).

Engine mechanisms beyond capture — salience, decay, supersession,
context-packing, feedback, and knowledge-graph edges — are deferred to later
phases.
"""

from __future__ import annotations

from typing import Any

from engram.config import Settings, get_settings
from engram.llm.client import QwenClient
from engram.memory.extraction import extract_memories
from engram.memory.models import Memory
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def embedding_text(memory: Memory) -> str:
    """Return the canonical text embedded for `memory`."""
    return f"{memory.type.value}: {memory.title}\n{memory.body}"


class MemoryEngine:
    """Coordinates extraction, the vector store, and the metadata store."""

    def __init__(
        self,
        client: QwenClient,
        vector_store: ChromaVectorStore,
        metadata_store: SqliteMetadataStore,
        settings: Settings | None,
    ) -> None:
        """Wire the engine to its client and stores and initialise the stores."""
        self._client = client
        self._vectors = vector_store
        self._meta = metadata_store
        self._settings = settings
        self._meta.init()
        self._vectors.init()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MemoryEngine:
        """Build an engine (client + both stores) from `settings`."""
        settings = settings or get_settings()
        client = QwenClient(settings)
        vector_store = ChromaVectorStore(path=settings.chroma_path)
        metadata_store = SqliteMetadataStore(path=settings.sqlite_path)
        return cls(client, vector_store, metadata_store, settings)

    def remember(
        self,
        content: str,
        project_id: str = "default",
        hint: str | None = None,
        source: str = "manual",
    ) -> list[Memory]:
        """Extract memories from `content`, store them, and return what was stored."""
        memories = extract_memories(self._client, content, project_id, source, hint)
        stored: list[Memory] = []
        for memory in memories:
            # Dedup: reuse the id of an existing same-key memory so it updates.
            existing = self._meta.find_by_key(project_id, memory.type.value, memory.title)
            if existing is not None:
                memory.id = existing.id
            embedding = self._client.embed([embedding_text(memory)])[0]
            self._meta.upsert_memory(memory)
            self._vectors.add_vector(
                memory.id,
                embedding,
                {"project_id": project_id, "type": memory.type.value},
            )
            stored.append(memory)
        return stored

    def recall(
        self, query: str, project_id: str = "default", k: int = 5
    ) -> list[dict[str, Any]]:
        """Return up to `k` memories most relevant to `query` within `project_id`."""
        embedding = self._client.embed([query])[0]
        hits = self._vectors.query(embedding, k, where={"project_id": project_id})
        distances = {hit_id: distance for hit_id, distance in hits}
        records = self._meta.get_memories(list(distances))
        results = [
            {
                "id": record.id,
                "type": record.type.value,
                "title": record.title,
                "body": record.body,
                "score": 1.0 - distances.get(record.id, 1.0),
            }
            for record in records
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def stats(self, project_id: str = "default") -> dict[str, Any]:
        """Return memory counts per type for `project_id` (used by `inspect`)."""
        by_type = self._meta.count_by_type(project_id)
        return {
            "project_id": project_id,
            "total": sum(by_type.values()),
            "by_type": by_type,
        }

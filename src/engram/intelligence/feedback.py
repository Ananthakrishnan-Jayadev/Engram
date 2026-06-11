"""Feedback loop: helpful/not-helpful signals adjust salience and log events."""

from __future__ import annotations

from engram.intelligence.decay import clamp01
from engram.memory.models import Memory
from engram.storage.metadata_store import SqliteMetadataStore

# How much one feedback event moves salience.
FEEDBACK_DELTA = 0.1


def apply_feedback(
    store: SqliteMetadataStore, memory_id: str, helpful: bool
) -> Memory | None:
    """Apply a feedback signal to a memory and log the event.

    helpful -> bump salience and refresh last access; not-helpful -> lower
    salience. Returns the updated memory, or None if it does not exist.
    """
    store.log_feedback(memory_id, helpful)
    memory = store.get_memory(memory_id)
    if memory is None:
        return None

    delta = FEEDBACK_DELTA if helpful else -FEEDBACK_DELTA
    store.update_salience(memory_id, clamp01(memory.salience + delta))
    if helpful:
        store.update_access(memory_id)
    return store.get_memory(memory_id)

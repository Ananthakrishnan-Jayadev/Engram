"""Single source of truth for the model names Engram uses.

Model choices live here (resolved from `Settings`) so call sites reference a
constant instead of hard-coding strings.
"""

from __future__ import annotations

from engram.config import get_settings


def flash_model() -> str:
    """High-frequency, low-latency chat model."""
    return get_settings().model_flash


def heavy_model() -> str:
    """Heavier reasoning model (e.g. supersession analysis)."""
    return get_settings().model_heavy


def embed_model() -> str:
    """Embedding model used for semantic retrieval."""
    return get_settings().model_embed

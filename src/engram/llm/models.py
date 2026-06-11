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


def heavy_model_for(client: object) -> str | None:
    """Resolve the heavy model from a client's settings, if available.

    Returns None when the client exposes no settings (e.g. a test fake), so the
    client falls back to its own default model — and no API key is required.
    """
    settings = getattr(client, "_settings", None)
    return getattr(settings, "model_heavy", None)

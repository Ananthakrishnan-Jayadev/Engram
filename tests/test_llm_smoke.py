"""Network-dependent LLM smoke test.

Skipped automatically when DASHSCOPE_API_KEY is unset so CI passes without a key.
"""

from __future__ import annotations

import os

import pytest

from engram.llm.client import QwenClient

pytestmark = pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not set; skipping live LLM smoke test.",
)


def test_chat_returns_text() -> None:
    """A flash chat call returns non-empty text."""
    client = QwenClient()
    reply = client.chat([{"role": "user", "content": "Reply with the single word: ok"}])
    assert isinstance(reply, str)
    assert reply.strip() != ""


def test_embed_returns_vectors() -> None:
    """Embedding one string returns one non-empty vector."""
    client = QwenClient()
    vectors = client.embed(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0

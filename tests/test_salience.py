"""Tests for salience scoring (LLM mocked)."""

from __future__ import annotations

from engram.intelligence.salience import score_memory
from engram.memory.models import Memory
from engram.memory.types import MemoryType


def _memory(mtype: MemoryType = MemoryType.BUG_FIX) -> Memory:
    """Build a minimal memory of the given type."""
    return Memory(id="m1", project_id="p1", type=mtype, title="t", body="b")


class JSONClient:
    """Client returning a fixed JSON salience object."""

    def __init__(self, payload: str) -> None:
        """Store the canned chat payload."""
        self.payload = payload

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return the canned payload."""
        return self.payload


def test_score_in_range_and_returns_rationale() -> None:
    """A valid LLM rating yields a salience in [0, 1] and a rationale string."""
    client = JSONClient('{"score": 0.8, "rationale": "reusable fix"}')
    score, rationale = score_memory(client, _memory())  # type: ignore[arg-type]
    assert 0.0 <= score <= 1.0
    assert isinstance(rationale, str) and rationale


def test_falls_back_to_base_prior_on_garbage() -> None:
    """Unparseable LLM output falls back to the per-type base prior."""
    client = JSONClient("not json at all")
    mem = _memory(MemoryType.ARCHITECTURE)
    score, rationale = score_memory(client, mem)  # type: ignore[arg-type]
    assert 0.0 <= score <= 1.0
    assert "base prior" in rationale


def test_out_of_range_llm_score_is_clamped() -> None:
    """An LLM score outside [0, 1] is clamped before blending."""
    client = JSONClient('{"score": 5.0, "rationale": "x"}')
    score, _ = score_memory(client, _memory())  # type: ignore[arg-type]
    assert 0.0 <= score <= 1.0

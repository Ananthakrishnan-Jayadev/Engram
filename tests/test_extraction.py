"""Tests for memory extraction (LLM mocked, no network)."""

from __future__ import annotations

from engram.memory.extraction import extract_memories
from engram.memory.types import MemoryType

CANNED_JSON = """[
  {
    "type": "bug_fix",
    "title": "Auth crash on expired token",
    "body": "decode() raised on a None token; added a null check first.",
    "details": {
      "symptom": "NoneType crash in auth.py",
      "root_cause": "expired token returned None",
      "fix": "guard with `if token is None` before decode"
    }
  },
  {
    "type": "convention",
    "title": "Type-hint everything",
    "body": "All public functions carry type hints.",
    "details": {}
  }
]"""


class FakeClient:
    """Minimal client whose chat returns a canned string."""

    def __init__(self, response: str) -> None:
        """Store the canned `response` and a call counter."""
        self.response = response
        self.chat_calls = 0

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return the canned response, counting calls."""
        self.chat_calls += 1
        return self.response


def test_extract_parses_into_memories() -> None:
    """The canned JSON parses into typed Memory objects."""
    client = FakeClient(CANNED_JSON)
    memories = extract_memories(client, "session log", "proj", "test")  # type: ignore[arg-type]

    assert len(memories) == 2
    assert memories[0].type is MemoryType.BUG_FIX
    assert memories[1].type is MemoryType.CONVENTION
    assert all(m.project_id == "proj" and m.source == "test" for m in memories)
    assert all(m.id for m in memories)


def test_bug_fix_details_survive() -> None:
    """Type-specific bug_fix details round-trip through extraction."""
    client = FakeClient(CANNED_JSON)
    memories = extract_memories(client, "session log", "proj", "test")  # type: ignore[arg-type]

    bug = memories[0]
    assert bug.details["root_cause"] == "expired token returned None"
    assert "guard" in bug.details["fix"]


def test_code_fences_are_stripped() -> None:
    """JSON wrapped in markdown fences still parses."""
    client = FakeClient(f"```json\n{CANNED_JSON}\n```")
    memories = extract_memories(client, "session log", "proj", "test")  # type: ignore[arg-type]
    assert len(memories) == 2


def test_invalid_entries_skipped() -> None:
    """Entries with unknown types or missing fields are dropped."""
    payload = (
        '[{"type": "not_a_type", "title": "x", "body": "y"}, '
        '{"type": "component", "title": "Parser", "body": "Parses input."}]'
    )
    client = FakeClient(payload)
    memories = extract_memories(client, "log", "proj", "test")  # type: ignore[arg-type]
    assert len(memories) == 1
    assert memories[0].type is MemoryType.COMPONENT


def test_parse_failure_returns_empty_after_retry() -> None:
    """Unparseable output triggers one retry, then returns []."""
    client = FakeClient("sorry, I cannot do that")
    memories = extract_memories(client, "log", "proj", "test")  # type: ignore[arg-type]
    assert memories == []
    assert client.chat_calls == 2

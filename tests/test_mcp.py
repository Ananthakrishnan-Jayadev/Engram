"""Tests that the MCP server registers its tools and returns expected shapes."""

from __future__ import annotations

import pytest

from engram.mcp import server
from engram.memory.models import Memory
from engram.memory.types import MemoryType


class FakeEngine:
    """Stand-in engine returning canned values for the MCP tools."""

    def remember(
        self, content: str, project_id: str = "default", hint: str | None = None,
        source: str = "mcp",
    ) -> list[Memory]:
        """Return one canned stored memory."""
        return [
            Memory(
                id="m1",
                project_id=project_id,
                type=MemoryType.BUG_FIX,
                title="Auth crash",
                body="Fixed null token.",
            )
        ]

    def recall(self, query: str, project_id: str = "default", k: int = 5) -> list[dict]:
        """Return one canned recall result."""
        return [
            {"id": "m1", "type": "bug_fix", "title": "Auth crash", "body": "b", "score": 0.9}
        ]

    def stats(self, project_id: str = "default") -> dict:
        """Return canned stats."""
        return {"project_id": project_id, "total": 1, "by_type": {"bug_fix": 1}}


def test_server_name() -> None:
    """The server is named 'engram'."""
    assert server.mcp.name == "engram"


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    """All Phase 1 tools are registered."""
    tools = await server.mcp.list_tools()
    names = {tool.name for tool in tools}
    assert {"bootstrap", "remember", "recall", "answer", "inspect"} <= names


def test_remember_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """remember returns a compact list of {type, title}."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.remember("some session content")
    assert out == [{"type": "bug_fix", "title": "Auth crash"}]


def test_recall_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """recall returns dicts with the expected keys."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.recall("how did I fix auth")
    assert out
    assert set(out[0]) == {"id", "type", "title", "body", "score"}


def test_inspect_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """inspect returns engine stats."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.inspect()
    assert out["total"] == 1
    assert out["by_type"] == {"bug_fix": 1}

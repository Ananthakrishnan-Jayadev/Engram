"""Tests that the MCP server registers its tools and returns expected shapes."""

from __future__ import annotations

from typing import Any

import pytest

from engram.mcp import server
from engram.memory.models import Memory
from engram.memory.types import MemoryType


class FakeEngine:
    """Stand-in engine returning canned values for the MCP tools."""

    def remember(
        self,
        content: str,
        project_id: str = "default",
        hint: str | None = None,
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

    def recall(
        self,
        query: str,
        project_id: str = "default",
        k: int = 5,
        pack: bool = False,
        token_budget: int = 1500,
    ) -> list[dict[str, Any]]:
        """Return one canned recall result."""
        return [{"id": "m1", "type": "bug_fix", "title": "Auth crash", "body": "b", "score": 0.9}]

    def answer(self, question: str, project_id: str = "default") -> dict[str, Any]:
        """Return a canned synthesized answer."""
        return {"answer": "validate before decode", "used_memory_ids": ["m1"]}

    def feedback(self, memory_id: str, helpful: bool) -> Memory:
        """Return a canned updated memory."""
        return Memory(
            id=memory_id,
            project_id="p1",
            type=MemoryType.BUG_FIX,
            title="t",
            body="b",
            salience=0.7,
        )

    def stats(self, project_id: str = "default") -> dict[str, Any]:
        """Return canned stats."""
        return {"project_id": project_id, "total": 1, "by_type": {"bug_fix": 1}}


def test_server_name() -> None:
    """The server is named 'engram'."""
    assert server.mcp.name == "engram"


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    """All Phase 2 tools are registered."""
    tools = await server.mcp.list_tools()
    names = {tool.name for tool in tools}
    expected = {"bootstrap", "sync", "remember", "recall", "answer", "inspect", "feedback"}
    assert expected <= names


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


def test_answer_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """answer returns a synthesized answer and used memory ids."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.answer("how do I handle expired tokens?")
    assert out["answer"]
    assert out["used_memory_ids"] == ["m1"]


def test_feedback_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """feedback returns an ok status with the updated salience."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.feedback("m1", helpful=True)
    assert out["status"] == "ok"
    assert out["salience"] == 0.7


def test_inspect_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """inspect returns engine stats."""
    monkeypatch.setattr(server, "_engine", FakeEngine())
    out = server.inspect()
    assert out["total"] == 1
    assert out["by_type"] == {"bug_fix": 1}

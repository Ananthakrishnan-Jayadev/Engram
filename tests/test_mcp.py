"""Tests that the MCP server constructs and registers its tools."""

from __future__ import annotations

import pytest

from engram.mcp.server import mcp


def test_server_name() -> None:
    """The server is named 'engram'."""
    assert mcp.name == "engram"


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    """All Phase 0 stub tools are registered."""
    tools = await mcp.list_tools()
    names = {tool.name for tool in tools}
    assert {"bootstrap", "remember", "recall", "answer", "inspect"} <= names

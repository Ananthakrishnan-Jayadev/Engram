"""FastMCP server for Engram.

Phase 0 registers the tool surface (`bootstrap`, `remember`, `recall`,
`answer`, `inspect`) as stubs that return placeholder dicts. The real memory
engine is wired up in later phases.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("engram")


@mcp.tool()
def bootstrap(project_path: str) -> dict[str, Any]:
    """Scan a project at `project_path` and build its initial typed memory.

    STUB (Phase 0): returns a placeholder. No scanning is performed yet.
    """
    return {"status": "stub", "tool": "bootstrap", "project_path": project_path}


@mcp.tool()
def remember(type: str, title: str, body: str) -> dict[str, Any]:
    """Store a new typed memory (`type`, `title`, `body`) about the project.

    STUB (Phase 0): returns a placeholder. Nothing is persisted yet.
    """
    return {"status": "stub", "tool": "remember", "type": type, "title": title}


@mcp.tool()
def recall(query: str, k: int = 5) -> dict[str, Any]:
    """Retrieve the `k` memories most relevant to `query`.

    STUB (Phase 0): returns a placeholder with no real results.
    """
    return {"status": "stub", "tool": "recall", "query": query, "k": k, "results": []}


@mcp.tool()
def answer(question: str) -> dict[str, Any]:
    """Answer `question` using recalled, verified project memory.

    STUB (Phase 0): returns a placeholder answer.
    """
    return {"status": "stub", "tool": "answer", "question": question, "answer": None}


@mcp.tool()
def inspect() -> dict[str, Any]:
    """Report engine state: memory counts, decay status, graph size.

    STUB (Phase 0): returns a placeholder snapshot.
    """
    return {"status": "stub", "tool": "inspect", "memories": 0, "edges": 0}


def main() -> None:
    """Run the Engram MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

"""FastMCP server for Engram.

Phase 1 wires `remember`, `recall`, and `inspect` to a `MemoryEngine` (the
extract → store → recall capture path). `bootstrap` and `answer` remain stubs
for later phases.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from engram.engine import MemoryEngine

mcp = FastMCP("engram")

# Lazily-constructed singleton engine (built at startup in main(); kept lazy so
# importing this module — e.g. in tests — does not require credentials/stores).
_engine: MemoryEngine | None = None


def get_engine() -> MemoryEngine:
    """Return the process-wide MemoryEngine, constructing it on first use."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine.from_settings()
    return _engine


@mcp.tool()
def bootstrap(project_path: str) -> dict[str, Any]:
    """Scan a project at `project_path` and build its initial typed memory.

    STUB (Phase 3): returns a placeholder. No scanning is performed yet.
    """
    return {"status": "stub", "tool": "bootstrap", "project_path": project_path}


@mcp.tool()
def remember(
    content: str, project_id: str = "default", hint: str | None = None
) -> list[dict[str, str]]:
    """Extract and store typed memories from `content`.

    Returns a compact list of {type, title} for each stored memory.
    """
    stored = get_engine().remember(content, project_id=project_id, hint=hint, source="mcp")
    return [{"type": m.type.value, "title": m.title} for m in stored]


@mcp.tool()
def recall(query: str, project_id: str = "default", k: int = 5) -> list[dict[str, Any]]:
    """Return the `k` memories most relevant to `query` within `project_id`."""
    return get_engine().recall(query, project_id=project_id, k=k)


@mcp.tool()
def answer(question: str) -> dict[str, Any]:
    """Answer `question` using recalled, verified project memory.

    STUB (Phase 3): returns a placeholder answer.
    """
    return {"status": "stub", "tool": "answer", "question": question, "answer": None}


@mcp.tool()
def inspect(project_id: str = "default") -> dict[str, Any]:
    """Report memory counts per type for `project_id`."""
    return get_engine().stats(project_id=project_id)


def main() -> None:
    """Construct the engine and run the Engram MCP server over stdio."""
    get_engine()
    mcp.run()


if __name__ == "__main__":
    main()

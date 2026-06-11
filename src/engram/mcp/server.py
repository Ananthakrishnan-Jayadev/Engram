"""FastMCP server for Engram.

Wires `bootstrap`, `sync`, `remember`, `recall`, `answer`, `inspect`, and
`feedback` to a `MemoryEngine`.
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
def bootstrap(project_path: str, project_id: str = "default") -> dict[str, Any]:
    """Scan a project's code, docs, and git history into initial memory.

    Returns {entities, memories_by_type, links}.
    """
    return get_engine().bootstrap(project_path, project_id=project_id)


@mcp.tool()
def sync(project_path: str, project_id: str = "default") -> dict[str, Any]:
    """Re-scan code and recheck memories whose linked entities changed.

    Returns {changed, removed, new, rechecked, superseded, flagged}.
    """
    return get_engine().sync_code(project_path, project_id=project_id)


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
def recall(
    query: str,
    project_id: str = "default",
    k: int = 5,
    pack: bool = False,
    token_budget: int = 1500,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return the `k` memories most relevant to `query` within `project_id`.

    With `pack=True`, also returns a packed context under `token_budget`.
    """
    return get_engine().recall(
        query, project_id=project_id, k=k, pack=pack, token_budget=token_budget
    )


@mcp.tool()
def answer(question: str, project_id: str = "default") -> dict[str, Any]:
    """Answer `question` from recalled project memory: {answer, used_memory_ids}."""
    return get_engine().answer(question, project_id=project_id)


@mcp.tool()
def feedback(memory_id: str, helpful: bool) -> dict[str, Any]:
    """Record a helpful/not-helpful signal for a memory and adjust its salience."""
    updated = get_engine().feedback(memory_id, helpful)
    if updated is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "ok", "memory_id": memory_id, "salience": updated.salience}


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

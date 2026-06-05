# Engram

Engram is a project-scoped, local-first memory engine for coding agents that doesn't just
remember — it forgets correctly. It connects to any MCP-compatible coding agent (Qwen Code,
Claude Code, Cursor, Copilot CLI), bootstraps an organized typed memory of a project
(architecture, conventions, components, bug fixes, rejected approaches, open threads), and keeps
that memory accurate over time by detecting superseded knowledge and decaying what's stale. The
flagship capability is *verified* bug-fix recall: "how did I fix this before?" returns a fix
checked against the current codebase before it's injected.

## Stack

- **Language:** Python 3.11+
- **Agent interface:** Official Python MCP SDK (FastMCP)
- **LLM:** Qwen via DashScope (OpenAI-compatible endpoint)
- **Vectors + semantic search:** Chroma (embedded, persistent)
- **Records / knowledge graph / engine state:** SQLite

## Install (for humans)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install with dev extras
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env             # then edit .env and set DASHSCOPE_API_KEY
```

## Run

```bash
# Foundation smoke test (verifies LLM, embeddings, Chroma, SQLite end-to-end)
python scripts/smoke.py

# Lint + tests
ruff check .
pytest -q

# Start the MCP server over stdio
engram-mcp
```

## Phase 0 status

This repository is at **Phase 0 — scaffolding only**. The following are *functional*:
config loading, the Qwen/DashScope client (chat + embeddings), the Chroma vector store, the
SQLite metadata store, the typed-memory data models, an MCP server that starts and registers
tools, the smoke script, basic tests, and CI.

The memory **engine logic is not implemented yet** — salience, decay, supersession,
context-packing, and feedback are stubbed (`NotImplementedError("Phase 2")`), the MCP tools
return placeholders, and the eval metrics raise `NotImplementedError("Phase 4")`.

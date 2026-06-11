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

# Phase 1 capture demo (live key): remember a bug-fix session, then recall it
python scripts/demo_capture.py
```

## Phase 1 usage — the capture path

Phase 1 implements **extract → store → recall**. The `MemoryEngine` reads raw content
(a chat log, a diff, notes), asks Qwen to extract typed memories, embeds and stores them,
and recalls them semantically.

```python
from engram.engine import MemoryEngine

engine = MemoryEngine.from_settings()

# Extract + store typed memories from a session snippet.
stored = engine.remember(
    "Fixed a NoneType crash in auth.py: get_token() returned None on an "
    "expired token; added a null-check before jwt.decode().",
    project_id="my-project",
)
for m in stored:
    print(m.type.value, "-", m.title)

# Recall semantically.
for hit in engine.recall("how did I fix the auth crash", project_id="my-project"):
    print(f"{hit['score']:.3f}  [{hit['type']}] {hit['title']}")

# Counts per type.
print(engine.stats(project_id="my-project"))
```

The same surface is exposed over MCP via the `remember`, `recall`, and `inspect` tools.

## Phase status

**Functional now (Phase 0 + Phase 1):** config loading, the Qwen/DashScope client
(chat + embeddings), the Chroma vector store (cosine, precomputed embeddings), the SQLite
metadata store (records + JSON `details` + batch/count/dedup lookups), typed-memory models,
memory **extraction**, the `MemoryEngine` capture path (`remember`/`recall`/`stats`), the
MCP `remember`/`recall`/`inspect` tools, the smoke + demo scripts, tests, and CI.

**Still stubbed:** the MCP `bootstrap` and `answer` tools (placeholders), all engine
mechanisms — salience, decay, supersession, context-packing, feedback, and knowledge-graph
edges — and the eval metrics (`NotImplementedError("Phase 4")`).

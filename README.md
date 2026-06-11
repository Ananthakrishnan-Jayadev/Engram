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

# Phase 2 hero-story demo (live key): supersession + decay + answer + feedback
python scripts/demo_engine.py
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

## Phase 2 usage — the intelligence engine

Phase 2 adds **salience**, **decay/forgetting**, **content-driven supersession**,
**context-packing**, **answer synthesis**, and a **feedback loop**. `remember` now scores each
memory's salience and, at write-time, detects when a new memory supersedes/contradicts/duplicates
existing ones (marking them `superseded`). `recall` ranks active memories by
`0.7*similarity + 0.3*effective_strength`, excludes superseded/forgotten entries, and reinforces
what it returns.

```python
from engram.engine import MemoryEngine

engine = MemoryEngine.from_settings()

# Capture an original fix, then a refactor that supersedes it.
engine.remember("Quick fix: try/except AttributeError around jwt.decode in auth.py", project_id="p")
engine.remember("Refactor: a require_token() guard validates tokens up front, "
                "superseding the try/except hack", project_id="p")

# Recall returns only the still-valid memory (decayed/superseded ones are excluded).
for hit in engine.recall("expired token handling", project_id="p"):
    print(f"{hit['combined']:.3f}  [{hit['type']}] {hit['title']}")

# Synthesize a grounded answer from packed memory.
out = engine.answer("how do I handle expired tokens?", project_id="p")
print(out["answer"], out["used_memory_ids"])

# Recall with a packed, token-bounded context.
packed = engine.recall("token handling", project_id="p", pack=True, token_budget=800)
print(packed["context"]["est_tokens"], packed["context"]["included_ids"])

# Feedback nudges salience; maintenance applies decay-driven status transitions.
engine.feedback(out["used_memory_ids"][0], helpful=True)
engine.maintenance(project_id="p")
```

MCP exposes `remember`, `recall` (with `pack`/`token_budget`), `answer`, `feedback`, and
`inspect`. Decay constants (half-lives, thresholds) live in one block in
`engram/intelligence/decay.py` so Phase 4 can learn them.

## Phase status

**Functional now (Phases 0–2):** config, the Qwen/DashScope client, the Chroma vector store
(cosine, precomputed embeddings), the SQLite metadata store (records, JSON `details`, status /
access tracking, feedback table, in-place `migrate()`), typed-memory models, extraction, and the
`MemoryEngine` — `remember` (salience + supersession), `recall` (decay-aware ranking + packing),
`answer` (recall → pack → synthesize), `feedback`, and `maintenance`. MCP tools: `remember`,
`recall`, `answer`, `feedback`, `inspect`.

**Still stubbed / deferred:** the MCP `bootstrap` tool and code-edit-driven graph supersession
(Phase 3), learned decay-rate tuning (Phase 4), the eval metrics
(`NotImplementedError("Phase 4")`), and scheduled background consolidation (Phase 6 — Phase 2
provides a callable `maintenance()` only). Answer does not yet verify a fix against current
source (`# TODO Phase 3`).

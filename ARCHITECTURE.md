# Architecture

Engram is a local-first memory engine that sits between a coding agent and a project's accumulated
knowledge. The agent talks to Engram over MCP (`bootstrap`, `remember`, `recall`, `answer`,
`inspect`); Engram turns project artifacts and developer interactions into *typed* memories,
embeds them with Qwen, retrieves them semantically, and — in later phases — scores their salience,
decays stale entries, and propagates supersession across a code-aware knowledge graph so recalled
knowledge stays accurate. Engram deliberately splits persistence across **two stores**: **Chroma**
holds vector embeddings for fast semantic search, while **SQLite** is the source of truth for
typed-memory records, knowledge-graph edges, and engine state. Vectors and structured/relational
data have different access patterns, so keeping them separate (both behind a single storage
interface) lets each use the right tool and makes the vector layer swappable for a managed service
in cloud mode.

## File tree

```
engram/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── ARCHITECTURE.md
├── .github/workflows/ci.yml
├── src/engram/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings Settings + cached get_settings()
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py             # QwenClient (chat + embeddings) over DashScope
│   │   └── models.py             # model-name constants
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py               # StorageInterface contract
│   │   ├── vector_store.py        # ChromaVectorStore (precomputed embeddings)
│   │   ├── metadata_store.py      # SqliteMetadataStore
│   │   └── schema.py             # SQLite DDL
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── types.py              # MemoryType enum
│   │   └── models.py             # Memory pydantic model
│   └── mcp/
│       ├── __init__.py
│       └── server.py             # FastMCP server + tool stubs
├── eval/
│   ├── __init__.py
│   ├── harness.py                # Benchmark skeleton + metric stubs
│   └── fixtures/sample_project/   # synthetic project + bug_history.json
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_storage.py
│   ├── test_llm_smoke.py
│   └── test_mcp.py
└── scripts/smoke.py
```

## Two-store rationale

- **Chroma (vectors):** purpose-built for approximate nearest-neighbour search over embeddings.
  Engram supplies *precomputed* Qwen embeddings rather than using Chroma's default embedding
  function, so the embedding model stays under Engram's control.
- **SQLite (records / graph / state):** a single-file, zero-server relational store for the
  authoritative memory records, knowledge-graph edges, salience/decay state, and feedback. It is
  transactional, easy to back up, and queryable with plain SQL.

Both layers implement a shared `StorageInterface` so callers depend on the contract, not the
backend — enabling a swap to a managed vector store in cloud mode without touching engine code.

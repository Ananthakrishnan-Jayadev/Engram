"""Foundation smoke test for Engram (run by a human).

Exercises every Phase 0 building block end-to-end:
  1. load settings
  2. chat with the flash model (expect "ok")
  3. embed a string and report its dimension
  4. Chroma round-trip with a real Qwen embedding
  5. SQLite init + insert/read one Memory

Each step is isolated in try/except so one failure does not hide the others.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

from engram.config import get_settings
from engram.llm.client import QwenClient
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore


def main() -> None:
    """Run all smoke steps and print a per-step and final summary."""
    results: dict[str, bool] = {}

    # Step 1: settings ----------------------------------------------------
    settings = None
    try:
        settings = get_settings()
        print(f"✅ 1. Settings loaded (base_url={settings.base_url})")
        results["settings"] = True
    except Exception as exc:  # noqa: BLE001 - smoke test reports every failure
        print(f"❌ 1. Settings failed: {exc}")
        results["settings"] = False

    client = QwenClient(settings) if settings else None

    # Step 2: chat --------------------------------------------------------
    try:
        assert client is not None, "client unavailable (settings failed)"
        reply = client.chat(
            [{"role": "user", "content": "Reply with exactly: ok"}]
        )
        print(f"✅ 2. Chat (flash) replied: {reply!r}")
        results["chat"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 2. Chat failed: {exc}")
        results["chat"] = False

    # Step 3: embeddings --------------------------------------------------
    embedding: list[float] | None = None
    try:
        assert client is not None, "client unavailable (settings failed)"
        vectors = client.embed(["hello"])
        embedding = vectors[0]
        print(f"✅ 3. Embedding dimension: {len(embedding)}")
        results["embed"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 3. Embedding failed: {exc}")
        results["embed"] = False

    # ignore_cleanup_errors: on Windows, Chroma keeps a file handle open which
    # makes the temp-dir cleanup raise PermissionError/WinError 32 on teardown.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        vstore: ChromaVectorStore | None = None

        # Step 4: Chroma round-trip --------------------------------------
        try:
            vector = embedding or [0.1, 0.2, 0.3]
            vstore = ChromaVectorStore(path=str(tmp_path / "chroma"))
            vstore.add_vector("smoke-1", vector, {"project_id": "smoke"})
            hits = vstore.query(vector, k=1)
            assert hits and hits[0]["id"] == "smoke-1", "round-trip mismatch"
            print(f"✅ 4. Chroma round-trip ok (got id={hits[0]['id']})")
            results["chroma"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 4. Chroma round-trip failed: {exc}")
            results["chroma"] = False

        # Step 5: SQLite insert/read -------------------------------------
        try:
            mstore = SqliteMetadataStore(path=str(tmp_path / "engram.sqlite"))
            mstore.init()
            mem = Memory(
                id="smoke-1",
                project_id="smoke",
                type=MemoryType.BUG_FIX,
                title="smoke memory",
                body="inserted by scripts/smoke.py",
            )
            mstore.upsert_memory(mem)
            fetched = mstore.get_memory("smoke-1")
            assert fetched is not None and fetched.title == "smoke memory"
            print(f"✅ 5. SQLite insert/read ok (title={fetched.title!r})")
            results["sqlite"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 5. SQLite insert/read failed: {exc}")
            results["sqlite"] = False

        # Release Chroma references so handles are freed before temp-dir
        # cleanup (best effort; cleanup errors are ignored regardless).
        vstore = None
        gc.collect()

    # Summary -------------------------------------------------------------
    passed = sum(results.values())
    total = len(results)
    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n{passed}/{total} steps passed.")


if __name__ == "__main__":
    main()

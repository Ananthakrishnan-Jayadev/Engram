"""End-to-end harness smoke test against real Qwen (skipped without a key)."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not set; skipping live e2e benchmark smoke test.",
)


def test_e2e_benchmark_smoke() -> None:
    """A tiny scenario runs end-to-end and Engram is no worse than NaiveAll on stale."""
    from engram.config import get_settings
    from engram.engine import MemoryEngine
    from engram.llm.client import QwenClient
    from engram.storage.metadata_store import SqliteMetadataStore
    from engram.storage.vector_store import ChromaVectorStore
    from eval.baselines import EngramStrategy, NaiveAll
    from eval.generator import generate_scenario
    from eval.harness import run_benchmark

    scenario = generate_scenario(seed=0, n_topics=1, supersession_prob=1.0, distractors=1)
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj = Path(tmp) / "proj"
        proj.mkdir()
        store = Path(tmp) / "store"
        store.mkdir()
        for path, source in scenario.files.items():
            (proj / path).write_text(source, encoding="utf-8")

        settings = get_settings()
        client = QwenClient(settings)
        engine = MemoryEngine(
            client=client,
            vector_store=ChromaVectorStore(path=str(store / "chroma")),
            metadata_store=SqliteMetadataStore(path=str(store / "db.sqlite")),
            settings=settings,
        )
        strategies = [
            NaiveAll(client.embed),
            EngramStrategy(
                engine, proj, project_id="bench", mode="e2e", transcripts=scenario.transcripts
            ),
        ]
        results = run_benchmark(scenario, strategies, k=5, mode="e2e")

        engram = results.per_strategy["engram"]
        assert 0.0 <= engram["recall"]["hit_at_k"] <= 1.0
        assert engram["stale_hit_rate"] <= results.per_strategy["naive_all"]["stale_hit_rate"]

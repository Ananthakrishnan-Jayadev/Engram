"""End-to-end (offline) harness test: Engram beats NaiveAll on stale_hit_rate."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from engram.engine import MemoryEngine
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore
from eval.baselines import EngramStrategy, NaiveAll
from eval.generator import generate_scenario
from eval.harness import run_benchmark
from tests.eval_fakes import FakeClient, fake_embed


def test_engram_beats_naive_on_stale_hit_rate() -> None:
    """On a tiny scenario, Engram retires the stale memory; NaiveAll keeps it."""
    scenario = generate_scenario(seed=2, n_topics=1, supersession_prob=1.0, distractors=1)
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proj = Path(tmp) / "proj"
        proj.mkdir()
        store = Path(tmp) / "store"
        store.mkdir()
        for path, source in scenario.files.items():
            (proj / path).write_text(source, encoding="utf-8")

        client: Any = FakeClient()
        engine = MemoryEngine(
            client=client,
            vector_store=ChromaVectorStore(path=str(store / "chroma")),
            metadata_store=SqliteMetadataStore(path=str(store / "db.sqlite")),
            settings=None,
        )
        strategies = [NaiveAll(fake_embed), EngramStrategy(engine, proj, project_id="bench")]
        results = run_benchmark(scenario, strategies, k=5)

        engram_stale = results.per_strategy["engram"]["stale_hit_rate"]
        naive_stale = results.per_strategy["naive_all"]["stale_hit_rate"]
        assert engram_stale < naive_stale
        assert results.per_strategy["engram"]["recall"]["hit_at_k"] == 1.0

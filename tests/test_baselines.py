"""Tests for benchmark strategies (offline fake LLM + fake embedder)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from engram.engine import MemoryEngine
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore
from eval.baselines import EngramStrategy, NaiveAll, NoMemory
from eval.generator import Scenario, generate_scenario
from tests.eval_fakes import FakeClient, fake_embed


def _build(tmp: str, scenario: Scenario) -> MemoryEngine:
    """Write scenario files to a project dir and build a fake-backed engine."""
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
    return engine


def _replay(strategy: Any, scenario: Scenario) -> None:
    """Replay all scenario events into a single strategy."""
    for event in scenario.events:
        if event.kind == "add" and event.memory is not None:
            strategy.add_memory(event.memory)
        elif event.kind == "edit" and event.edit is not None:
            strategy.apply_edit(event.edit)


def test_baselines_handle_stale_differently() -> None:
    """NoMemory is empty; NaiveAll surfaces the stale memory; Engram retires it."""
    scenario = generate_scenario(seed=1, n_topics=1, supersession_prob=1.0, distractors=1)
    query = scenario.queries[0]
    assert query.stale_keys  # the scenario actually has a superseded memory

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = _build(tmp, scenario)

        nomem = NoMemory()
        naive = NaiveAll(fake_embed)
        engram = EngramStrategy(engine, Path(tmp) / "proj", project_id="bench")
        for strategy in (nomem, naive, engram):
            _replay(strategy, scenario)

        assert nomem.query(query.text, 5) == []

        naive_keys = naive.query(query.text, 5)
        assert any(stale in naive_keys for stale in query.stale_keys)

        engram_keys = engram.query(query.text, 5)
        assert not any(stale in engram_keys for stale in query.stale_keys)
        assert query.gold_key in engram_keys
        assert query.stale_keys[0] in engram.retired_keys()

"""Learned forgetting = fit the deterministic policy to benchmark outcomes.

This is NOT a trained model. We ingest a scenario once via the real engine (so
real supersession decisions are made once and frozen), then coordinate-search
the deterministic policy parameters (half-lives, decay thresholds, ranking
weights, packing budget) over the labelled queries to maximise

    objective = recall_accuracy - stale_hit_rate

re-ranking the frozen store with no further LLM calls. Cheap and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.engine import RECALL_FETCH_BUFFER
from engram.intelligence.decay import effective_strength, next_status
from engram.intelligence.policy import ForgettingPolicy
from eval.baselines import EngramStrategy
from eval.generator import Scenario
from eval.metrics import QueryOutcome, recall_accuracy, stale_hit_rate

if TYPE_CHECKING:
    from engram.engine import MemoryEngine
    from engram.memory.models import Memory

# Deterministic coordinate-search grids.
_W_SIM_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
_HALFLIFE_SCALE: tuple[float, ...] = (0.5, 1.0, 2.0)
_DORMANT_GRID: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30)
_FORGOTTEN_GRID: tuple[float, ...] = (0.02, 0.05, 0.10)
_BUDGET_GRID: tuple[int, ...] = (1000, 1500, 2000)


@dataclass
class FrozenQuery:
    """A query plus its (memory_id, distance) hits captured once from the store."""

    gold_key: str
    stale_keys: list[str]
    hits: list[tuple[str, float]]


@dataclass
class FrozenStore:
    """A snapshot enabling LLM-free re-ranking under candidate policies."""

    now: datetime
    records: dict[str, Memory]  # memory id -> record
    key_by_id: dict[str, str]  # memory id -> ground-truth key
    queries: list[FrozenQuery] = field(default_factory=list)


def objective(frozen: FrozenStore, policy: ForgettingPolicy, k: int = 5) -> float:
    """Return recall_accuracy - stale_hit_rate under `policy` (no LLM calls)."""
    outcomes: list[QueryOutcome] = []
    for fq in frozen.queries:
        scored: list[tuple[float, str]] = []
        for memory_id, distance in fq.hits:
            record = frozen.records.get(memory_id)
            if record is None or record.status != "active":
                continue
            strength = effective_strength(record, frozen.now, policy)
            age_days = max(0.0, (frozen.now - record.created_at).total_seconds() / 86400.0)
            if next_status(strength, age_days, policy) != "active":
                continue
            similarity = 1.0 - distance
            combined = policy.w_sim * similarity + policy.w_str * strength
            scored.append((combined, memory_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        returned = [frozen.key_by_id.get(mid, mid) for _, mid in scored[:k]]
        outcomes.append(QueryOutcome(fq.gold_key, fq.stale_keys, returned))
    return recall_accuracy(outcomes, k)["hit_at_k"] - stale_hit_rate(outcomes, k)


def _scaled_half_lives(base: dict[str, float], scale: float) -> dict[str, float]:
    """Scale every per-type half-life by `scale`."""
    return {key: value * scale for key, value in base.items()}


def search_policy(
    frozen: FrozenStore, k: int = 5, base: ForgettingPolicy | None = None
) -> tuple[ForgettingPolicy, float, float]:
    """Deterministic coordinate search; returns (best_policy, before, after).

    `before` is the objective of the default policy; `after` is the best found.
    Each dimension is optimised greedily in a fixed order, so the result is fully
    reproducible.
    """
    default = ForgettingPolicy.default()
    before = objective(frozen, default, k)
    best = base or default
    best_obj = objective(frozen, best, k)

    def _try(candidate: ForgettingPolicy) -> None:
        nonlocal best, best_obj
        value = objective(frozen, candidate, k)
        if value > best_obj:
            best, best_obj = candidate, value

    for w_sim in _W_SIM_GRID:
        _try(replace(best, w_sim=w_sim, w_str=round(1.0 - w_sim, 3)))
    for scale in _HALFLIFE_SCALE:
        _try(
            replace(
                best,
                half_life_days=_scaled_half_lives(best.half_life_days, scale),
                default_half_life_days=best.default_half_life_days * scale,
            )
        )
    for dormant in _DORMANT_GRID:
        _try(replace(best, dormant_threshold=dormant))
    for forgotten in _FORGOTTEN_GRID:
        _try(replace(best, forgotten_threshold=forgotten))
    for budget in _BUDGET_GRID:
        _try(replace(best, token_budget=budget))

    return best, before, best_obj


def freeze_store(
    engine: MemoryEngine, strategy: EngramStrategy, scenario: Scenario, k: int, project_id: str
) -> FrozenStore:
    """Capture the post-ingest store: query hits (one embed each) + records."""
    now = datetime.now(UTC)
    records = {m.id: m for m in engine._meta.all_memories(project_id)}
    fetch = k + RECALL_FETCH_BUFFER
    queries: list[FrozenQuery] = []
    for query in scenario.queries:
        embedding = engine._client.embed([query.text])[0]
        hits = engine._vectors.query(embedding, fetch, where={"project_id": project_id})
        queries.append(FrozenQuery(query.gold_key, query.stale_keys, hits))
    return FrozenStore(
        now=now, records=records, key_by_id=dict(strategy._key_by_id), queries=queries
    )


def fit_policy(
    scenario: Scenario,
    engine: MemoryEngine,
    project_dir: str | Path,
    project_id: str = "fit",
    mode: str = "direct",
    k: int = 5,
    runs: int = 1,
    out_path: str | Path | None = None,
) -> tuple[ForgettingPolicy, dict[str, Any]]:
    """Ingest once, freeze the store, search the policy, and (optionally) write it.

    `runs` is reserved; ingestion happens once (real supersession is frozen).
    """
    strategy = EngramStrategy(
        engine, project_dir, project_id=project_id, mode=mode, transcripts=scenario.transcripts
    )
    for event in scenario.events:
        if event.kind == "add" and event.memory is not None:
            strategy.add_memory(event.memory)
        elif event.kind == "edit" and event.edit is not None:
            strategy.apply_edit(event.edit)

    frozen = freeze_store(engine, strategy, scenario, k, project_id)
    fitted, before, after = search_policy(frozen, k)

    report: dict[str, Any] = {
        "objective_before": before,
        "objective_after": after,
        "default": ForgettingPolicy.default().to_dict(),
        "fitted": fitted.to_dict(),
    }
    if out_path is not None:
        path = Path(out_path)
        fitted.save(path)
        summary = (
            "# Learned forgetting (fit to benchmark outcomes)\n\n"
            f"- seed: {scenario.seed}\n"
            f"- objective (recall - stale_hit): before={before:.3f}, after={after:.3f}\n\n"
            "Learned = the deterministic policy params were searched against the "
            "labelled queries over a frozen store. Not a trained model.\n"
        )
        path.with_suffix(".md").write_text(summary, encoding="utf-8")
    return fitted, report

"""Tests for fit-to-outcomes policy search on a fixed frozen store (no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime

from engram.intelligence.policy import ForgettingPolicy
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from eval.tuning import FrozenQuery, FrozenStore, objective, search_policy


def _mem(mid: str, salience: float, now: datetime) -> Memory:
    """Build an active, freshly-accessed memory with the given salience."""
    return Memory(
        id=mid,
        project_id="p",
        type=MemoryType.BUG_FIX,
        title=mid,
        body="b",
        salience=salience,
        status="active",
        created_at=now,
        last_accessed=now,
        access_count=0,
    )


def _frozen() -> FrozenStore:
    """A store where similarity alone ranks the wrong memory first.

    "g" (gold) has lower similarity but high salience/strength; "n" has higher
    similarity but low strength. At the default weights, "n" wins; shifting
    weight toward strength recovers "g".
    """
    now = datetime(2026, 1, 1, tzinfo=UTC)
    records = {"g": _mem("g", 0.9, now), "n": _mem("n", 0.3, now)}
    return FrozenStore(
        now=now,
        records=records,
        key_by_id={"g": "g", "n": "n"},
        queries=[FrozenQuery(gold_key="g", stale_keys=[], hits=[("g", 0.4), ("n", 0.05)])],
    )


def test_search_improves_objective_over_default() -> None:
    """The coordinate search finds a policy better than default() on this store."""
    frozen = _frozen()
    before = objective(frozen, ForgettingPolicy.default(), k=1)
    best, reported_before, after = search_policy(frozen, k=1)

    assert reported_before == before
    assert after > before  # there is room, and the search uses it


def test_search_is_deterministic() -> None:
    """Same frozen store + fixed grids => identical fitted policy and objective."""
    frozen = _frozen()
    best_a, _, after_a = search_policy(frozen, k=1)
    best_b, _, after_b = search_policy(frozen, k=1)
    assert after_a == after_b
    assert (best_a.w_sim, best_a.w_str) == (best_b.w_sim, best_b.w_str)

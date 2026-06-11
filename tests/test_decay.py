"""Tests for decay strength and status transitions (pure, no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from engram.intelligence import decay
from engram.intelligence.decay import effective_strength, next_status
from engram.memory.models import Memory
from engram.memory.types import MemoryType


def _memory(**kw: object) -> Memory:
    """Build a Memory with sensible defaults for decay tests."""
    base = {
        "id": "m1",
        "project_id": "p1",
        "type": MemoryType.BUG_FIX,
        "title": "t",
        "body": "b",
        "salience": 1.0,
    }
    base.update(kw)
    return Memory(**base)  # type: ignore[arg-type]


def test_strength_drops_with_age() -> None:
    """Strength decreases as time since last access grows."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fresh = _memory(created_at=now)
    old = _memory(created_at=now - timedelta(days=60))

    assert effective_strength(fresh, now) > effective_strength(old, now)


def test_access_boost_raises_strength() -> None:
    """More accesses raise strength for the same age."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    last = now - timedelta(days=10)
    rarely = _memory(last_accessed=last, access_count=0)
    often = _memory(last_accessed=last, access_count=20)

    assert effective_strength(often, now) > effective_strength(rarely, now)


def test_strength_in_unit_range() -> None:
    """Strength is always clamped to [0, 1]."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    mem = _memory(created_at=now, salience=1.0, access_count=1000)
    value = effective_strength(mem, now)
    assert 0.0 <= value <= 1.0


def test_status_thresholds() -> None:
    """next_status maps strength + age to active/dormant/forgotten."""
    assert next_status(0.9, age_days=1.0) == "active"
    assert next_status(0.10, age_days=1.0) == "dormant"
    # Below the forgotten threshold but still within the grace period -> dormant.
    assert next_status(0.01, age_days=1.0) == "dormant"
    # Below the forgotten threshold and past the grace period -> forgotten.
    assert next_status(0.01, age_days=decay.GRACE_PERIOD_DAYS + 1) == "forgotten"

"""Spaced-repetition decay and forgetting — pure functions, no LLM.

All tunable constants live in one block so Phase 4 can learn them.
"""

from __future__ import annotations

from datetime import datetime
from math import log

from engram.memory.models import Memory
from engram.memory.types import MemoryType

# --- Tunable constants (Phase 4 will learn these) ------------------------

# Per-type half-life in days: how long until decay halves the strength.
HALF_LIFE_DAYS: dict[MemoryType, float] = {
    MemoryType.OPEN_THREAD: 3.0,
    MemoryType.BUG_FIX: 30.0,
    MemoryType.COMPONENT: 60.0,
    MemoryType.REJECTED_APPROACH: 90.0,
    MemoryType.CONVENTION: 120.0,
    MemoryType.ARCHITECTURE: 180.0,
}
DEFAULT_HALF_LIFE_DAYS = 30.0

# Recall reinforcement: each access nudges strength up logarithmically.
ACCESS_BOOST_COEFF = 0.1

# Status thresholds on effective strength.
DORMANT_THRESHOLD = 0.15
FORGOTTEN_THRESHOLD = 0.05

# A memory must be at least this old before it can be "forgotten".
GRACE_PERIOD_DAYS = 14.0


def clamp01(value: float) -> float:
    """Clamp `value` into the [0, 1] range."""
    return max(0.0, min(1.0, value))


def half_life_for(memory_type: MemoryType) -> float:
    """Return the half-life in days for `memory_type`."""
    return HALF_LIFE_DAYS.get(memory_type, DEFAULT_HALF_LIFE_DAYS)


def _days_between(later: datetime, earlier: datetime) -> float:
    """Return non-negative whole-and-fractional days between two datetimes."""
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def effective_strength(memory: Memory, now: datetime) -> float:
    """Compute a memory's current strength in [0, 1] given `now`.

    strength = clamp(salience * decay * access_boost), where decay is a
    spaced-repetition half-life curve over time since last access.
    """
    last_access = memory.last_accessed or memory.created_at
    days_since = _days_between(now, last_access)
    half_life = half_life_for(memory.type)
    decay = 0.5 ** (days_since / half_life)
    access_boost = 1.0 + ACCESS_BOOST_COEFF * log(1.0 + memory.access_count)
    return clamp01(memory.salience * decay * access_boost)


def next_status(strength: float, age_days: float) -> str:
    """Map `strength` (and age) to a decay status: active/dormant/forgotten."""
    if strength < FORGOTTEN_THRESHOLD and age_days >= GRACE_PERIOD_DAYS:
        return "forgotten"
    if strength < DORMANT_THRESHOLD:
        return "dormant"
    return "active"

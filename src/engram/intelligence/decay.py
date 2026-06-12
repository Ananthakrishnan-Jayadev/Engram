"""Spaced-repetition decay and forgetting — pure functions, no LLM.

Tunable constants live in `ForgettingPolicy`; these functions read from a policy
instance (defaulting to `DEFAULT_POLICY`), so behaviour is unchanged at defaults.
"""

from __future__ import annotations

from datetime import datetime
from math import log

from engram.intelligence.policy import ForgettingPolicy
from engram.memory.models import Memory

# Source of truth for defaults is the policy; these module constants are kept as
# convenience aliases for callers/tests that reference them directly.
DEFAULT_POLICY = ForgettingPolicy.default()
DORMANT_THRESHOLD = DEFAULT_POLICY.dormant_threshold
FORGOTTEN_THRESHOLD = DEFAULT_POLICY.forgotten_threshold
GRACE_PERIOD_DAYS = DEFAULT_POLICY.grace_period_days


def clamp01(value: float) -> float:
    """Clamp `value` into the [0, 1] range."""
    return max(0.0, min(1.0, value))


def _days_between(later: datetime, earlier: datetime) -> float:
    """Return non-negative whole-and-fractional days between two datetimes."""
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def effective_strength(
    memory: Memory, now: datetime, policy: ForgettingPolicy | None = None
) -> float:
    """Compute a memory's current strength in [0, 1] given `now`.

    strength = clamp(salience * decay * access_boost), where decay is a
    spaced-repetition half-life curve over time since last access.
    """
    policy = policy or DEFAULT_POLICY
    last_access = memory.last_accessed or memory.created_at
    days_since = _days_between(now, last_access)
    half_life = policy.half_life_for(memory.type)
    decay = 0.5 ** (days_since / half_life)
    access_boost = 1.0 + policy.access_boost_coeff * log(1.0 + memory.access_count)
    return clamp01(memory.salience * decay * access_boost)


def next_status(
    strength: float, age_days: float, policy: ForgettingPolicy | None = None
) -> str:
    """Map `strength` (and age) to a decay status: active/dormant/forgotten."""
    policy = policy or DEFAULT_POLICY
    if strength < policy.forgotten_threshold and age_days >= policy.grace_period_days:
        return "forgotten"
    if strength < policy.dormant_threshold:
        return "dormant"
    return "active"

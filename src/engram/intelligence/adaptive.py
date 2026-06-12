"""OPTIONAL STRETCH (off by default, not wired in): online forgetting adaptation.

Phase 4 ships *fit-to-outcomes* learned forgetting (see eval/tuning.py): a cheap,
reproducible search over the deterministic policy. This module sketches a
different, online idea — nudging per-type half-lives from observed re-access
intervals and feedback as the system runs — so the update rule is on record. It
is deliberately NOT used by the engine; enable it consciously if ever desired.

Update rule (sketch)
---------------------
For a memory type t, let the observed re-access interval be the time between
consecutive recalls of memories of that type. If memories of type t are recalled
again sooner than their half-life predicts, the half-life is too long (we are
keeping them strong past their usefulness window) — and vice versa. Nudge:

    half_life[t] <- clamp(half_life[t] * (1 + lr * signal))

where `signal` is positive when observed intervals exceed the half-life
(decay too fast) and negative when they fall short (decay too slow), and helpful
feedback gently lengthens the half-life while not-helpful shortens it.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.intelligence.policy import ForgettingPolicy

# This stub is inert unless explicitly turned on.
ADAPTIVE_ENABLED = False


@dataclass
class AdaptiveConfig:
    """Tuning knobs for the (disabled) online half-life adaptation."""

    learning_rate: float = 0.05
    min_half_life_days: float = 1.0
    max_half_life_days: float = 365.0


def adapt_half_lives(
    policy: ForgettingPolicy,
    observed_interval_days: dict[str, float],
    config: AdaptiveConfig | None = None,
) -> ForgettingPolicy:
    """Return a policy with per-type half-lives nudged toward observed intervals.

    STRETCH STUB: returns `policy` unchanged unless `ADAPTIVE_ENABLED` is set.
    The body documents the intended update; it is not used by the engine.
    """
    if not ADAPTIVE_ENABLED:
        return policy

    config = config or AdaptiveConfig()
    updated = dict(policy.half_life_days)
    for mtype, half_life in updated.items():
        observed = observed_interval_days.get(mtype)
        if observed is None or half_life <= 0:
            continue
        # signal > 0 when memories outlive their half-life (decay too fast).
        signal = (observed - half_life) / half_life
        nudged = half_life * (1.0 + config.learning_rate * signal)
        updated[mtype] = max(config.min_half_life_days, min(config.max_half_life_days, nudged))

    from dataclasses import replace

    return replace(policy, half_life_days=updated)

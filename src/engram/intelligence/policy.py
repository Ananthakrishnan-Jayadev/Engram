"""ForgettingPolicy: all tunable forgetting/ranking constants in one place.

`default()` returns the current hand-set values, so behaviour is unchanged
unless a fitted policy is loaded (Phase 4 tuning). JSON load/save make a fitted
policy reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ForgettingPolicy:
    """Tunable parameters for decay, ranking, packing, and supersession."""

    half_life_days: dict[str, float]  # keyed by MemoryType value
    default_half_life_days: float
    access_boost_coeff: float
    dormant_threshold: float
    forgotten_threshold: float
    grace_period_days: float
    w_sim: float
    w_str: float
    token_budget: int
    supersede_min_confidence: float

    @classmethod
    def default(cls) -> ForgettingPolicy:
        """Return the current hand-set defaults (behaviour unchanged)."""
        return cls(
            half_life_days={
                "open_thread": 3.0,
                "bug_fix": 30.0,
                "component": 60.0,
                "rejected_approach": 90.0,
                "convention": 120.0,
                "architecture": 180.0,
            },
            default_half_life_days=30.0,
            access_boost_coeff=0.1,
            dormant_threshold=0.15,
            forgotten_threshold=0.05,
            grace_period_days=14.0,
            w_sim=0.7,
            w_str=0.3,
            token_budget=1500,
            supersede_min_confidence=0.7,
        )

    def half_life_for(self, memory_type: Any) -> float:
        """Return the half-life in days for a memory type (enum or value)."""
        key = getattr(memory_type, "value", memory_type)
        return self.half_life_days.get(key, self.default_half_life_days)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this policy."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write this policy to `path` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ForgettingPolicy:
        """Load a policy from a JSON file written by `save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

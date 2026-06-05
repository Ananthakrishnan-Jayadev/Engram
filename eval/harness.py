"""Benchmark harness for measuring Engram's memory quality.

All logic is deferred to Phase 4; this module defines the shape of the
benchmark and its metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Benchmark:
    """Runs a synthetic-project benchmark against an Engram engine.

    The benchmark loads a fixture project (source plus a known bug history),
    drives an engine through a session, and scores the resulting memory along
    four axes (see the metric methods).
    """

    def load_fixture(self, path: str | Path) -> Any:
        """Load a fixture project (source files + bug_history.json) from `path`."""
        raise NotImplementedError("Phase 4")

    def run(self, engine: Any) -> Any:
        """Drive `engine` through the benchmark and collect raw outcomes."""
        raise NotImplementedError("Phase 4")

    def recall_accuracy(self) -> float:
        """Fraction of queries for which the correct memory was retrieved."""
        raise NotImplementedError("Phase 4")

    def forgetting_correctness(self) -> float:
        """How well the engine decays/supersedes stale memories without
        dropping still-valid ones."""
        raise NotImplementedError("Phase 4")

    def packing_efficiency(self) -> float:
        """Useful information per token in the context packed for the agent."""
        raise NotImplementedError("Phase 4")

    def improvement_curve(self) -> Any:
        """Agent-performance trend across sessions as memory accumulates."""
        raise NotImplementedError("Phase 4")

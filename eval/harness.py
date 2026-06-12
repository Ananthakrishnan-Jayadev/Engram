"""Benchmark harness: replay a scenario into strategies and score them.

Replaces the Phase-0 skeleton. `mode="direct"` injects structured memories (no
extraction variance); `mode="e2e"` (Part 2) drives real Qwen extraction.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eval.baselines import Strategy
from eval.generator import Scenario
from eval.metrics import (
    QueryOutcome,
    forgetting_prf,
    recall_accuracy,
    recheck_prf,
    stale_hit_rate,
)

StrategyFactory = Callable[[], list[Strategy]]

# Scalar metric paths aggregated by run_repeated (dotted into per-strategy dicts).
_SCALAR_PATHS: tuple[str, ...] = (
    "recall.hit_at_k",
    "recall.mrr",
    "stale_hit_rate",
    "forgetting.f1",
    "recheck.f1",
    "packing.token_ratio",
    "packing.gold_retention",
)


@dataclass
class Results:
    """Per-strategy metric bundle for one benchmark run."""

    seed: int
    k: int
    mode: str
    per_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)


# How deep to look when diagnosing where the gold memory ended up.
DIAG_K = 50


def _evaluate(strategy: Strategy, scenario: Scenario, k: int) -> list[QueryOutcome]:
    """Run every query against `strategy` at its current state."""
    return [
        QueryOutcome(q.gold_key, q.stale_keys, strategy.query(q.text, k)) for q in scenario.queries
    ]


def _gold_diagnostics(strategy: Strategy, scenario: Scenario, k: int) -> list[dict[str, Any]]:
    """Per-query gold-rank breakdown explaining hit@k.

    For each query: the gold memory's rank (or None) and why it missed —
    `outside_top_k` (present but ranked too low) or `excluded:<status>` (not
    retrievable, e.g. superseded/forgotten).
    """
    records: list[dict[str, Any]] = []
    for query in scenario.queries:
        ranking = strategy.query(query.text, DIAG_K)
        if query.gold_key in ranking:
            rank = ranking.index(query.gold_key) + 1
            hit = rank <= k
            reason = "hit" if hit else "outside_top_k"
        else:
            rank = None
            hit = False
            reason = f"excluded:{strategy.status_of(query.gold_key)}"
        records.append(
            {
                "query": query.text,
                "gold_key": query.gold_key,
                "rank": rank,
                "hit": hit,
                "reason": reason,
            }
        )
    return records


def run_benchmark(
    scenario: Scenario,
    strategies: list[Strategy],
    k: int = 5,
    mode: str = "direct",
    progress: Callable[[str], None] | None = None,
) -> Results:
    """Replay `scenario` into each strategy and compute all metrics.

    `progress`, if given, receives short status strings (ingestion + querying)
    so callers can surface live progress.
    """
    emit = progress or (lambda _message: None)
    truth_stale = set(scenario.supersession_truth.keys())
    recheck_affected = {key for keys in scenario.recheck_truth.values() for key in keys}
    checkpoints = set(scenario.checkpoints)
    total = len(scenario.events)
    step = max(1, total // 5)

    per_strategy: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        emit(f"Ingesting {strategy.name} ({total} events)")
        curve: list[float] = []
        for i, event in enumerate(scenario.events):
            if event.kind == "add" and event.memory is not None:
                strategy.add_memory(event.memory)
            elif event.kind == "edit" and event.edit is not None:
                strategy.apply_edit(event.edit)
            if (i + 1) % step == 0 or (i + 1) == total:
                emit(f"  ingested {i + 1}/{total}")
            if (i + 1) in checkpoints:
                outcomes = _evaluate(strategy, scenario, k)
                curve.append(recall_accuracy(outcomes, k)["hit_at_k"])

        emit(f"Querying {strategy.name} ({len(scenario.queries)} queries)")
        outcomes = _evaluate(strategy, scenario, k)
        metrics: dict[str, Any] = {
            "recall": recall_accuracy(outcomes, k),
            "stale_hit_rate": stale_hit_rate(outcomes, k),
            "forgetting": forgetting_prf(strategy.retired_keys(), truth_stale),
            "recheck": recheck_prf(
                strategy.flagged_keys() | strategy.retired_keys(), recheck_affected
            ),
            "curve": curve,
        }
        packing = getattr(strategy, "packing_stats", None)
        if callable(packing):
            metrics["packing"] = packing(scenario, k)
        metrics["gold_miss"] = _gold_diagnostics(strategy, scenario, k)
        per_strategy[strategy.name] = metrics

    return Results(seed=scenario.seed, k=k, mode=mode, per_strategy=per_strategy)


@dataclass
class AggregateResults:
    """Per-strategy, per-metric mean ± std across repeated runs."""

    seed: int
    k: int
    mode: str
    runs: int
    per_strategy: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _dig(metrics: dict[str, Any], dotted: str) -> float | None:
    """Walk a dotted path into a nested metric dict; return a number or None."""
    node: Any = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


def run_repeated(
    scenario: Scenario,
    strategy_factory: StrategyFactory,
    runs: int = 5,
    k: int = 5,
    mode: str = "e2e",
    progress: Callable[[str], None] | None = None,
) -> AggregateResults:
    """Run the benchmark `runs` times (fresh strategies each) and aggregate.

    `strategy_factory` must return a *fresh* set of strategies per call, since a
    strategy carries state across a run (a single set cannot be reused).
    """
    emit = progress or (lambda _message: None)
    runs = max(1, runs)
    results = []
    for i in range(runs):
        emit(f"Run {i + 1}/{runs}")
        results.append(
            run_benchmark(scenario, strategy_factory(), k=k, mode=mode, progress=progress)
        )

    names = list(results[0].per_strategy)
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        per_metric: dict[str, dict[str, float]] = {}
        for path in _SCALAR_PATHS:
            values = [
                v for r in results if (v := _dig(r.per_strategy.get(name, {}), path)) is not None
            ]
            if not values:
                continue
            per_metric[path] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        aggregate[name] = per_metric

    return AggregateResults(seed=scenario.seed, k=k, mode=mode, runs=runs, per_strategy=aggregate)

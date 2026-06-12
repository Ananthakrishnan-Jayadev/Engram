"""Tests for the deterministic scenario generator."""

from __future__ import annotations

from eval.generator import generate_scenario


def test_same_seed_is_identical() -> None:
    """The same seed produces an identical scenario."""
    assert generate_scenario(7) == generate_scenario(7)


def test_labels_are_internally_consistent() -> None:
    """Supersession, query, and recheck labels reference real, correctly-tagged keys."""
    scenario = generate_scenario(3)
    specs = {e.memory.key: e.memory for e in scenario.events if e.kind == "add"}

    for stale_key, superseding_key in scenario.supersession_truth.items():
        assert specs[stale_key].label == "stale"
        assert specs[superseding_key].label == "current"

    for query in scenario.queries:
        assert query.gold_key in specs
        assert specs[query.gold_key].label == "current"
        assert query.gold_key not in query.stale_keys
        for stale_key in query.stale_keys:
            assert stale_key in scenario.supersession_truth

    for affected in scenario.recheck_truth.values():
        for key in affected:
            assert key in specs


def test_checkpoints_within_event_range() -> None:
    """Checkpoints index real points in the event stream."""
    scenario = generate_scenario(5)
    assert scenario.checkpoints
    assert all(1 <= cp <= len(scenario.events) for cp in scenario.checkpoints)

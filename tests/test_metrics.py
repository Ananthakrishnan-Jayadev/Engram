"""Tests for benchmark metrics on hand-built inputs."""

from __future__ import annotations

from eval.metrics import (
    QueryOutcome,
    forgetting_prf,
    improvement_curve,
    packing_efficiency,
    recall_accuracy,
    recheck_prf,
    stale_hit_rate,
)


def test_recall_accuracy_hit_and_mrr() -> None:
    """hit@k and MRR match a hand-computed example."""
    outcomes = [
        QueryOutcome("a", ["b"], ["a", "c"]),  # gold rank 1
        QueryOutcome("x", ["y"], ["y", "x"]),  # gold rank 2
    ]
    result = recall_accuracy(outcomes, k=2)
    assert result["hit_at_k"] == 1.0
    assert abs(result["mrr"] - 0.75) < 1e-9


def test_stale_hit_rate() -> None:
    """Only the second query surfaces a stale key in the top-2."""
    outcomes = [
        QueryOutcome("a", ["b"], ["a", "c"]),
        QueryOutcome("x", ["y"], ["y", "x"]),
    ]
    assert stale_hit_rate(outcomes, k=2) == 0.5


def test_forgetting_prf() -> None:
    """Retirement precision/recall/F1 against supersession truth."""
    assert forgetting_prf({"b"}, {"b": "c"}) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    noisy = forgetting_prf({"b", "z"}, {"b": "c"})
    assert noisy["precision"] == 0.5
    assert noisy["recall"] == 1.0
    assert abs(noisy["f1"] - (2 * 0.5 * 1.0 / 1.5)) < 1e-9


def test_recheck_prf() -> None:
    """Recheck PRF against the union of edit-affected keys."""
    result = recheck_prf({"m1"}, {"e1": ["m1", "m2"]})
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_packing_efficiency_and_curve() -> None:
    """Packing ratio/retention and the identity improvement curve."""
    packing = packing_efficiency(50, 100, 2, 4)
    assert packing["token_ratio"] == 0.5
    assert packing["gold_retention"] == 0.5
    assert improvement_curve([0.1, 0.2]) == [0.1, 0.2]

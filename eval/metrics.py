"""Benchmark metrics — pure functions over hand-buildable inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class QueryOutcome:
    """One query's gold answer, stale keys to avoid, and ranked returned keys."""

    gold_key: str
    stale_keys: list[str]
    returned: list[str]


def recall_accuracy(outcomes: list[QueryOutcome], k: int) -> dict[str, float]:
    """Return hit@k (gold in top-k) and MRR (over the full ranking)."""
    if not outcomes:
        return {"hit_at_k": 0.0, "mrr": 0.0}
    hits = 0
    rr = 0.0
    for outcome in outcomes:
        if outcome.gold_key in outcome.returned[:k]:
            hits += 1
        if outcome.gold_key in outcome.returned:
            rr += 1.0 / (outcome.returned.index(outcome.gold_key) + 1)
    n = len(outcomes)
    return {"hit_at_k": hits / n, "mrr": rr / n}


def stale_hit_rate(outcomes: list[QueryOutcome], k: int) -> float:
    """Fraction of queries with any stale key in the top-k (lower is better)."""
    if not outcomes:
        return 0.0
    bad = 0
    for outcome in outcomes:
        topk = set(outcome.returned[:k])
        if any(stale in topk for stale in outcome.stale_keys):
            bad += 1
    return bad / len(outcomes)


def _prf(predicted: set[str], truth: set[str]) -> dict[str, float]:
    """Precision/recall/F1 of `predicted` against `truth` (vacuously 1.0 empty)."""
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def forgetting_prf(
    retired: Iterable[str], supersession_truth: dict[str, str] | Iterable[str]
) -> dict[str, float]:
    """PRF of retired memories vs the keys that should have been retired."""
    truth = (
        set(supersession_truth.keys())
        if isinstance(supersession_truth, dict)
        else set(supersession_truth)
    )
    return _prf(set(retired), truth)


def recheck_prf(
    surfaced: Iterable[str], recheck_truth: dict[str, list[str]] | Iterable[str]
) -> dict[str, float]:
    """PRF of flagged/retired memories vs the truly edit-affected keys."""
    if isinstance(recheck_truth, dict):
        truth = {key for keys in recheck_truth.values() for key in keys}
    else:
        truth = set(recheck_truth)
    return _prf(set(surfaced), truth)


def packing_efficiency(
    packed_tokens: int, naive_tokens: int, gold_in_pack: int, total_gold: int
) -> dict[str, float]:
    """Packed vs naive token cost, and the fraction of gold retained in the pack."""
    return {
        "packed_tokens": float(packed_tokens),
        "naive_tokens": float(naive_tokens),
        "token_ratio": (packed_tokens / naive_tokens) if naive_tokens else 1.0,
        "gold_retention": (gold_in_pack / total_gold) if total_gold else 1.0,
    }


def improvement_curve(checkpoint_accuracies: Iterable[float]) -> list[float]:
    """The recall-accuracy trajectory across checkpoints."""
    return list(checkpoint_accuracies)

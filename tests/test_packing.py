"""Tests for context packing (no LLM; client omitted)."""

from __future__ import annotations

from engram.intelligence.packing import estimate_tokens, pack


def _memory(mid: str, combined: float, body: str) -> dict[str, object]:
    """Build a recall-result dict for packing."""
    return {
        "id": mid,
        "type": "bug_fix",
        "title": f"title {mid}",
        "body": body,
        "combined": combined,
    }


def test_never_exceeds_budget() -> None:
    """The packed context's estimated tokens never exceed the budget."""
    body = "word " * 50
    memories = [_memory(f"m{i}", combined=0.5, body=body) for i in range(10)]
    packed = pack(memories, token_budget=60, query="q")
    assert packed.est_tokens <= 60


def test_orders_by_value() -> None:
    """Higher-value memories are selected/ordered ahead of lower-value ones."""
    body = "same length body here"  # identical token cost across items
    memories = [
        _memory("low", combined=0.1, body=body),
        _memory("high", combined=0.9, body=body),
        _memory("mid", combined=0.5, body=body),
    ]
    packed = pack(memories, token_budget=1000, query="q")
    assert packed.included_ids[0] == "high"
    assert packed.included_ids.index("high") < packed.included_ids.index("mid")
    assert packed.included_ids.index("mid") < packed.included_ids.index("low")


def test_estimate_tokens_heuristic() -> None:
    """Token estimate follows the len//4 heuristic with a floor of 1."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 40) == 10

"""Context packing: greedily select recalled memories under a token budget."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from engram.llm.client import QwenClient
from engram.prompts import COMPRESSION_SYSTEM, build_compression_prompt

# Compress a memory body once its block exceeds this estimated token count.
COMPRESS_THRESHOLD_TOKENS = 120


class PackedContext(BaseModel):
    """A budget-bounded context assembled from recalled memories."""

    text: str
    included_ids: list[str]
    est_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate token count for `text`.

    TODO Phase 4: swap this len//4 heuristic for a real tokenizer.
    """
    return max(1, len(text) // 4)


def _format_block(memory: dict[str, Any], body: str | None = None) -> str:
    """Render one memory as a context block."""
    return f"[{memory['type']}] {memory['title']}\n{body if body is not None else memory['body']}"


def _compress(client: QwenClient, body: str, query: str) -> str:
    """Compress `body` to its load-bearing slice; fall back to `body` on error."""
    messages = [
        {"role": "system", "content": COMPRESSION_SYSTEM},
        {"role": "user", "content": build_compression_prompt(body, query)},
    ]
    try:
        out = client.chat(messages).strip()
    except Exception:  # noqa: BLE001 - compression is best-effort
        return body
    return out or body


def _value_of(memory: dict[str, Any]) -> float:
    """Combined recall score used as the memory's packing value."""
    return float(memory.get("combined", memory.get("score", 0.0)))


def pack(
    memories: list[dict[str, Any]],
    token_budget: int,
    query: str,
    client: QwenClient | None = None,
) -> PackedContext:
    """Greedily pack `memories` by value-per-token, never exceeding `token_budget`.

    Long bodies may be compressed via Qwen flash when a `client` is supplied.
    """
    scored: list[tuple[float, int, str, str]] = []
    for memory in memories:
        block = _format_block(memory)
        tokens = estimate_tokens(block)
        if client is not None and tokens > COMPRESS_THRESHOLD_TOKENS:
            compressed = _compress(client, memory["body"], query)
            block = _format_block(memory, body=compressed)
            tokens = estimate_tokens(block)
        value = _value_of(memory)
        scored.append((value, tokens, block, memory["id"]))

    # Greedy by value-per-token; skip blocks that would overflow the budget.
    scored.sort(key=lambda item: item[0] / item[1], reverse=True)

    blocks: list[str] = []
    included_ids: list[str] = []
    total = 0
    for _value, tokens, block, memory_id in scored:
        if total + tokens > token_budget:
            continue
        blocks.append(block)
        included_ids.append(memory_id)
        total += tokens

    return PackedContext(text="\n\n".join(blocks), included_ids=included_ids, est_tokens=total)

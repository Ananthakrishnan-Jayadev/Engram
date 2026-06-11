"""Salience scoring: how durably useful a memory is, in [0, 1]."""

from __future__ import annotations

from engram.intelligence.decay import clamp01
from engram.llm.client import QwenClient
from engram.llm.parsing import request_json
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.prompts import SALIENCE_SYSTEM, build_salience_prompt

# Per-type base prior: durable knowledge starts high, transient starts low.
BASE_PRIOR: dict[MemoryType, float] = {
    MemoryType.ARCHITECTURE: 0.90,
    MemoryType.CONVENTION: 0.85,
    MemoryType.COMPONENT: 0.70,
    MemoryType.BUG_FIX: 0.60,
    MemoryType.REJECTED_APPROACH: 0.50,
    MemoryType.OPEN_THREAD: 0.35,
}
DEFAULT_PRIOR = 0.5

# Weight on the type prior vs. the LLM reusability rating.
PRIOR_WEIGHT = 0.5


def score_memory(client: QwenClient, memory: Memory) -> tuple[float, str]:
    """Return a salience in [0, 1] plus a one-line rationale for `memory`.

    Blends a per-type base prior with a Qwen flash reusability rating. Falls
    back to the base prior if the LLM rating is missing or unparseable.
    """
    prior = BASE_PRIOR.get(memory.type, DEFAULT_PRIOR)
    messages = [
        {"role": "system", "content": SALIENCE_SYSTEM},
        {
            "role": "user",
            "content": build_salience_prompt(memory.type.value, memory.title, memory.body),
        },
    ]
    data = request_json(client, messages)

    if isinstance(data, dict) and isinstance(data.get("score"), int | float):
        llm_score = clamp01(float(data["score"]))
        rationale = str(data.get("rationale", "")).strip() or "llm reusability rating"
        salience = clamp01(PRIOR_WEIGHT * prior + (1.0 - PRIOR_WEIGHT) * llm_score)
        return salience, rationale

    return prior, "base prior (LLM rating unavailable)"

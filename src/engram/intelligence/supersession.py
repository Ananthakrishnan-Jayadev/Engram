"""Content-driven supersession: does a new memory change existing ones?

Phase 2 compares at write-time only. Code-edit-driven supersession via the
knowledge graph is Phase 3.
"""

from __future__ import annotations

from pydantic import BaseModel

from engram.intelligence.decay import clamp01
from engram.llm.client import QwenClient
from engram.llm.models import heavy_model_for
from engram.llm.parsing import request_json
from engram.memory.models import Memory
from engram.prompts import SUPERSESSION_SYSTEM, build_supersession_prompt

RELATIONS = {"supersedes", "contradicts", "duplicate", "unrelated"}


class Verdict(BaseModel):
    """A judgement about how a new memory relates to one existing candidate."""

    target_id: str
    relation: str
    confidence: float
    rationale: str = ""


def _format_candidates(candidates: list[Memory]) -> str:
    """Render candidates as an id-tagged block for the prompt."""
    return "\n".join(
        f"- id: {c.id}\n"
        f"  type: {c.type.value}\n"
        f"  title: {c.title}\n"
        f"  body: {c.body}"
        for c in candidates
    )


def check_supersession(
    client: QwenClient, new_memory: Memory, candidates: list[Memory]
) -> list[Verdict]:
    """Compare `new_memory` against `candidates` with one heavy LLM call.

    Returns one validated Verdict per candidate the model judged. Invalid or
    unknown entries are dropped; on parse failure it returns [].
    """
    if not candidates:
        return []

    messages = [
        {"role": "system", "content": SUPERSESSION_SYSTEM},
        {
            "role": "user",
            "content": build_supersession_prompt(
                new_memory.type.value,
                new_memory.title,
                new_memory.body,
                _format_candidates(candidates),
            ),
        },
    ]
    data = request_json(client, messages, model=heavy_model_for(client))
    if not isinstance(data, list):
        return []

    valid_ids = {c.id for c in candidates}
    verdicts: list[Verdict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        target_id = item.get("target_id")
        relation = item.get("relation")
        if target_id not in valid_ids or relation not in RELATIONS:
            continue
        try:
            confidence = clamp01(float(item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        verdicts.append(
            Verdict(
                target_id=target_id,
                relation=relation,
                confidence=confidence,
                rationale=str(item.get("rationale", "")),
            )
        )
    return verdicts

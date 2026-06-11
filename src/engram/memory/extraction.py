"""Extract typed memories from raw content using the LLM."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from engram.llm.client import QwenClient
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.prompts import (
    EXTRACTION_SYSTEM,
    STRICT_RETRY,
    build_extraction_prompt,
)

_VALID_TYPES: set[str] = {t.value for t in MemoryType}


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from `text`."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _parse_items(raw: str) -> list[Any] | None:
    """Parse `raw` into a JSON list, or return None on failure.

    Tolerates code fences and surrounding prose by also trying the substring
    between the first '[' and last ']'.
    """
    text = _strip_fences(raw)
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            return data
    return None


def _coerce_items(items: list[Any], project_id: str, source: str) -> list[Memory]:
    """Convert raw JSON items into validated `Memory` objects, skipping invalid."""
    memories: list[Memory] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        type_value = item.get("type")
        if type_value not in _VALID_TYPES:
            continue
        title = item.get("title")
        body = item.get("body")
        if not isinstance(title, str) or not isinstance(body, str):
            continue
        details = item.get("details")
        if not isinstance(details, dict):
            details = {}
        entities = item.get("entities")
        if isinstance(entities, list):
            hints = [e.strip() for e in entities if isinstance(e, str) and e.strip()]
            if hints:
                details = {**details, "entity_hints": hints}
        memories.append(
            Memory(
                id=uuid.uuid4().hex,
                project_id=project_id,
                type=MemoryType(type_value),
                title=title.strip(),
                body=body.strip(),
                source=source,
                details=details,
            )
        )
    return memories


def extract_memories(
    client: QwenClient,
    content: str,
    project_id: str,
    source: str,
    hint: str | None = None,
) -> list[Memory]:
    """Extract typed memories from `content` via the LLM (flash model).

    Parses a strict JSON array; invalid entries are skipped. On a parse failure
    it retries once with a stricter instruction, then returns [].
    """
    user_prompt = build_extraction_prompt(content, hint)
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    # chat() defaults to the flash model.
    items = _parse_items(client.chat(messages))
    if items is None:
        retry_messages = [*messages, {"role": "user", "content": STRICT_RETRY}]
        items = _parse_items(client.chat(retry_messages))
    if items is None:
        return []
    return _coerce_items(items, project_id, source)

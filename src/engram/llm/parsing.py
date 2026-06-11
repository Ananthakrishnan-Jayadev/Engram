"""Robust JSON parsing for LLM replies, plus a retry-once request helper."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

_STRICT_JSON_RETRY = (
    "Your previous reply was not valid JSON. Reply with ONLY the JSON value "
    "(object or array) — no prose, no markdown fences."
)


class _ChatClient(Protocol):
    """Minimal client surface the helpers below need."""

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return the assistant's text content."""
        ...


def strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from `text`."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()


def parse_json(raw: str) -> Any | None:
    """Parse `raw` into JSON, tolerating fences and surrounding prose.

    Returns the parsed value, or None if nothing parseable is found.
    """
    text = strip_fences(raw)
    candidates = [text]
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start, end = text.find(open_char), text.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def request_json(
    client: _ChatClient,
    messages: list[dict[str, str]],
    model: str | None = None,
    retry_message: str = _STRICT_JSON_RETRY,
) -> Any | None:
    """Chat for a JSON reply; on parse failure retry once with a stricter ask."""
    data = parse_json(client.chat(messages, model=model))
    if data is None:
        retry = [*messages, {"role": "user", "content": retry_message}]
        data = parse_json(client.chat(retry, model=model))
    return data

"""Prompt templates for memory extraction.

Kept in one place so the wording is easy to iterate without touching logic.
"""

from __future__ import annotations

EXTRACTION_SYSTEM = (
    "You extract durable, reusable project knowledge from developer notes, chat "
    "logs, diffs, and code. You output ONLY a JSON array — no prose, no markdown "
    "code fences, no explanation."
)

_EXTRACTION_TEMPLATE = """\
Read the CONTENT below and extract the distinct, reusable memories worth keeping
about this project. Ignore transient chatter and anything not durably useful.

Return ONLY a JSON array. Each element must be an object with these keys:
  - "type": one of exactly: architecture, convention, component, bug_fix,
    rejected_approach, open_thread
  - "title": a short, specific label (a few words)
  - "body": a concise 1-3 sentence summary
  - "details": a type-specific object. For "bug_fix" use
    {{"symptom": ..., "root_cause": ..., "fix": ...}}. For other types use a
    small object with any relevant fields, or {{}} if none.

Rules:
  - Output the JSON array and nothing else (no ```json fences, no commentary).
  - If nothing is worth remembering, output [].
{hint_block}
CONTENT:
{content}
"""

STRICT_RETRY = (
    "Your previous reply was not valid JSON. Reply with ONLY a JSON array of "
    "memory objects, nothing else — no prose, no markdown fences."
)


def build_extraction_prompt(content: str, hint: str | None = None) -> str:
    """Render the extraction user prompt for `content` with an optional `hint`."""
    hint_block = f"\nHINT (focus on this): {hint}\n" if hint else ""
    return _EXTRACTION_TEMPLATE.format(hint_block=hint_block, content=content)

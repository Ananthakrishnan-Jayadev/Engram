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
  - "entities": a list of strings naming the files or code symbols this memory
    concerns (e.g. "auth.py", "TokenStore", "authenticate"). Use [] if unsure.

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


# --- Salience ------------------------------------------------------------

SALIENCE_SYSTEM = (
    "You rate how reusable and durably useful a project memory is for future "
    "coding work. You output ONLY a JSON object — no prose, no fences."
)

_SALIENCE_TEMPLATE = """\
Rate the long-term reusability of this memory on a 0.0-1.0 scale, where 1.0 is
knowledge a developer will repeatedly need (durable architecture, conventions,
verified fixes) and 0.0 is throwaway/transient.

Return ONLY: {{"score": <float 0..1>, "rationale": "<one short line>"}}

MEMORY
  type:  {type}
  title: {title}
  body:  {body}
"""


def build_salience_prompt(type_: str, title: str, body: str) -> str:
    """Render the salience (reusability) rating prompt for a memory."""
    return _SALIENCE_TEMPLATE.format(type=type_, title=title, body=body)


# --- Supersession --------------------------------------------------------

SUPERSESSION_SYSTEM = (
    "You decide whether a NEW project memory changes the status of existing "
    "memories. You output ONLY a JSON array — no prose, no fences."
)

_SUPERSESSION_TEMPLATE = """\
A NEW memory has just been recorded. For each CANDIDATE below, decide its
relation to the NEW memory and how the NEW memory affects it.

relation must be one of:
  - "supersedes": the NEW memory replaces the SAME specific fact/decision the
    candidate states (e.g. a newer version of the same behaviour)
  - "contradicts": the NEW memory directly conflicts with the candidate's claim
  - "duplicate": they express essentially the same knowledge
  - "unrelated": no meaningful relationship

Be strict: sharing a topic is NOT enough. If they cover the same topic but a
DIFFERENT aspect or granularity — e.g. a specific detail/note vs a general
description, or two complementary facts that can both be true — the relation is
"unrelated", not "supersedes". Only use "supersedes"/"contradicts" when the NEW
memory genuinely replaces or conflicts with the candidate's specific claim.

Return ONLY a JSON array; one object per candidate you have an opinion on:
  {{"target_id": "<candidate id>", "relation": "<relation>",
    "confidence": <float 0..1>, "rationale": "<one short line>"}}

NEW MEMORY
  type:  {new_type}
  title: {new_title}
  body:  {new_body}

CANDIDATES
{candidates}
"""


def build_supersession_prompt(new_type: str, new_title: str, new_body: str, candidates: str) -> str:
    """Render the supersession comparison prompt."""
    return _SUPERSESSION_TEMPLATE.format(
        new_type=new_type,
        new_title=new_title,
        new_body=new_body,
        candidates=candidates,
    )


# --- Packing compression -------------------------------------------------

COMPRESSION_SYSTEM = (
    "You compress a project memory to its load-bearing essence for an agent's "
    "context window. Output ONLY the condensed text — no preamble."
)

_COMPRESSION_TEMPLATE = """\
Condense the MEMORY BODY below to the minimum that still answers the QUERY,
keeping concrete specifics (symptoms, root causes, fixes, names). One or two
sentences. Output only the condensed text.

QUERY: {query}

MEMORY BODY:
{body}
"""


def build_compression_prompt(body: str, query: str) -> str:
    """Render the compression prompt for a memory body relative to `query`."""
    return _COMPRESSION_TEMPLATE.format(body=body, query=query)


# --- Answer synthesis ----------------------------------------------------

ANSWER_SYSTEM = (
    "You are a coding assistant that answers using ONLY the provided project "
    "memory. You synthesize a clear, grounded flow and never invent facts."
)

_ANSWER_TEMPLATE = """\
Using only the CONTEXT below, synthesize a direct, actionable answer to the
QUESTION. Prefer a short step-by-step flow. If the context is insufficient, say
so plainly. Do not invent details that are not in the context.

QUESTION: {question}

CONTEXT:
{context}
"""


def build_answer_prompt(question: str, context: str) -> str:
    """Render the answer-synthesis prompt from recalled `context`."""
    return _ANSWER_TEMPLATE.format(question=question, context=context)


# --- Code recheck (sync) -------------------------------------------------

CODE_RECHECK_SYSTEM = (
    "You decide whether a stored project memory is still valid given the "
    "current source of the code it concerns. You output ONLY a JSON object."
)

_CODE_RECHECK_TEMPLATE = """\
A memory is linked to a code entity whose source may have changed (or been
removed). Decide whether the memory still holds for the CURRENT SOURCE.

relation must be one of:
  - "still_valid": the memory is still accurate
  - "needs_update": mostly right, but some specific detail is now stale
  - "outdated": the memory's specific claim is now factually FALSE and it
    should be retired

Be conservative. PREFER "needs_update" over "outdated". Only choose "outdated"
when the memory's concrete claim is contradicted by the current source. An
implementation change *inside* a function (renamed locals, refactored body,
tweaked arithmetic) does NOT invalidate an architecture, convention, or
coupling claim unless the change actually removes or alters the described
relationship. When unsure, choose "needs_update", not "outdated".

Return ONLY: {{"relation": "<relation>", "confidence": <float 0..1>,
  "rationale": "<one short line>"}}

MEMORY
  type:  {mem_type}
  title: {mem_title}
  body:  {mem_body}

ENTITY: {entity_key}
CURRENT SOURCE:
{source}
"""


def build_code_recheck_prompt(
    mem_type: str, mem_title: str, mem_body: str, entity_key: str, source: str
) -> str:
    """Render the recheck prompt comparing a memory against current source."""
    return _CODE_RECHECK_TEMPLATE.format(
        mem_type=mem_type,
        mem_title=mem_title,
        mem_body=mem_body,
        entity_key=entity_key,
        source=source,
    )

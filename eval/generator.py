"""Deterministic synthetic-scenario generator for the Engram benchmark.

No LLM. Authoring the evolution of a project (memories added, superseded, and
code edited) yields exact ground-truth labels for every metric.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Marker tokens the deterministic offline fakes key on (see tests/eval_fakes.py).
# A superseding memory's body contains SUPERSEDE_MARKER; a stale (v0) body
# contains STALE_MARKER. Real Qwen ignores them.
SUPERSEDE_MARKER = "SUPERSEDES"
STALE_MARKER = "version 0"

_TOPIC_WORDS = ("auth", "cache", "billing", "search", "upload", "render", "queue", "config")
_TYPES = ("architecture", "convention", "component", "bug_fix")


@dataclass
class MemorySpec:
    """A pre-structured memory to inject, with its ground-truth label."""

    key: str
    type: str
    title: str
    body: str
    details: dict[str, str]
    entity_hints: list[str]
    label: str  # "current" | "stale"


@dataclass
class CodeEdit:
    """An edit to one entity's source, with the memory keys it should affect."""

    entity_key: str
    path: str
    new_source: str
    affected_keys: list[str]


@dataclass
class Event:
    """An ordered scenario step: add a memory, or apply a code edit."""

    index: int
    kind: str  # "add" | "edit"
    memory: MemorySpec | None = None
    edit: CodeEdit | None = None


@dataclass
class Query:
    """A retrieval probe with its gold answer and the stale keys to avoid."""

    text: str
    topic: int
    gold_key: str
    stale_keys: list[str]


@dataclass
class Scenario:
    """A fully-labelled synthetic project evolution."""

    seed: int
    events: list[Event]
    queries: list[Query]
    supersession_truth: dict[str, str]  # stale key -> superseding key
    recheck_truth: dict[str, list[str]]  # entity_key -> affected memory keys
    checkpoints: list[int]  # event counts at which to evaluate the curve
    transcripts: dict[str, str] = field(default_factory=dict)  # key -> session text
    files: dict[str, str] = field(default_factory=dict)  # path -> source


def _v0_body(topic: str, func: str) -> str:
    """Body for the original (later possibly stale) memory."""
    return f"The {topic} module {func}() does X. ({STALE_MARKER} behaviour)."


def _v1_body(topic: str, func: str) -> str:
    """Body for the superseding memory."""
    return f"The {topic} module {func}() now does Y. {SUPERSEDE_MARKER} the earlier behaviour."


def _topic_tag(topic_idx: int) -> str:
    """A unique, readable topic tag (word for the first cycle, suffixed after)."""
    word = _TOPIC_WORDS[topic_idx % len(_TOPIC_WORDS)]
    return word if topic_idx < len(_TOPIC_WORDS) else f"{word}{topic_idx}"


def generate_scenario(
    seed: int,
    n_topics: int = 8,
    sessions: int = 6,
    supersession_prob: float = 0.6,
    distractors: int = 3,
) -> Scenario:
    """Generate a deterministic, fully-labelled benchmark scenario.

    Each topic contributes an evolving core memory (v0, sometimes superseded by
    v1) plus `sessions // 3` detail-note memories, each with its own query, so a
    default 8-topic / 6-session run yields roughly 20-30 labelled queries. Pass
    small `n_topics`/`sessions` for quick smoke runs.
    """
    rng = random.Random(seed)
    events: list[Event] = []
    queries: list[Query] = []
    supersession_truth: dict[str, str] = {}
    recheck_truth: dict[str, list[str]] = {}
    transcripts: dict[str, str] = {}
    files: dict[str, str] = {}
    notes_per_topic = max(0, sessions // 3)

    def _add(spec: MemorySpec) -> None:
        events.append(Event(index=len(events), kind="add", memory=spec))

    for topic_idx in range(n_topics):
        tag = _topic_tag(topic_idx)
        mtype = _TYPES[topic_idx % len(_TYPES)]
        path = f"{tag}.py"
        func = f"handle_{tag}"
        entity_key = f"{path}::{func}"
        files[path] = f"def {func}():\n    return {topic_idx}\n"
        topic_keys: list[str] = []  # active memories linked to this topic's func

        k0 = f"{tag}-v0"
        spec0 = MemorySpec(
            key=k0,
            type=mtype,
            title=f"{tag} behaviour v0",
            body=_v0_body(tag, func),
            details={},
            entity_hints=[func],
            label="current",
        )
        _add(spec0)
        transcripts[k0] = f"Session: implemented {func} in {path}. {spec0.body}"
        current_key = k0
        stale_keys: list[str] = []

        if rng.random() < supersession_prob:
            k1 = f"{tag}-v1"
            spec1 = MemorySpec(
                key=k1,
                type=mtype,
                title=f"{tag} behaviour v1",
                body=_v1_body(tag, func),
                details={},
                entity_hints=[func],
                label="current",
            )
            _add(spec1)
            transcripts[k1] = f"Session: refactored {func} in {path}. {spec1.body}"
            spec0.label = "stale"
            supersession_truth[k0] = k1
            current_key = k1
            stale_keys = [k0]

        topic_keys.append(current_key)
        queries.append(
            Query(
                text=f"how does {tag} {func} work now?",
                topic=topic_idx,
                gold_key=current_key,
                stale_keys=stale_keys,
            )
        )

        for s in range(notes_per_topic):
            note_key = f"{tag}-note-{s}"
            note_body = f"The {tag} module {func}() also handles detail case {s}; keep it in mind."
            note_spec = MemorySpec(
                key=note_key,
                type="component",
                title=f"{tag} detail {s}",
                body=note_body,
                details={},
                entity_hints=[func],
                label="current",
            )
            _add(note_spec)
            transcripts[note_key] = f"Session {s}: {note_body}"
            topic_keys.append(note_key)
            queries.append(
                Query(
                    text=f"what does {tag} {func} do for detail case {s}?",
                    topic=topic_idx,
                    gold_key=note_key,
                    stale_keys=[],
                )
            )

        if rng.random() < 0.5:
            edit = CodeEdit(
                entity_key=entity_key,
                path=path,
                new_source=f"def {func}():\n    return {topic_idx} + 100\n",
                affected_keys=list(topic_keys),
            )
            events.append(Event(index=len(events), kind="edit", edit=edit))
            recheck_truth[entity_key] = list(topic_keys)

    for d in range(distractors):
        key = f"distractor-{d}"
        spec = MemorySpec(
            key=key,
            type="open_thread",
            title=f"misc note {d}",
            body=f"Unrelated scratch note number {d}.",
            details={},
            entity_hints=[],
            label="current",
        )
        _add(spec)
        transcripts[key] = f"Session: {spec.body}"

    n = len(events)
    checkpoints = sorted({max(1, n // 2), n})
    return Scenario(
        seed=seed,
        events=events,
        queries=queries,
        supersession_truth=supersession_truth,
        recheck_truth=recheck_truth,
        checkpoints=checkpoints,
        transcripts=transcripts,
        files=files,
    )

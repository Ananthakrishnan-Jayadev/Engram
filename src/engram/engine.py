"""MemoryEngine: the intelligence engine v1.

Orchestrates the capture path (extract → store → recall) plus salience scoring,
content-driven supersession, decay-aware ranking, context-packing, answer
synthesis, the feedback loop, and a callable maintenance pass.

Deferred: bootstrap (Phase 3), code-edit-driven supersession via the knowledge
graph (Phase 3), learned decay-rate tuning (Phase 4), scheduled background
consolidation (Phase 6 — this module exposes a callable maintenance() only).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from engram.config import Settings, get_settings
from engram.intelligence.decay import effective_strength, next_status
from engram.intelligence.feedback import apply_feedback
from engram.intelligence.packing import PackedContext, pack
from engram.intelligence.salience import score_memory
from engram.intelligence.supersession import check_supersession
from engram.llm.client import QwenClient
from engram.llm.models import heavy_model_for
from engram.memory.extraction import extract_memories
from engram.memory.models import Memory
from engram.memory.types import MemoryType
from engram.prompts import ANSWER_SYSTEM, build_answer_prompt
from engram.storage.metadata_store import SqliteMetadataStore
from engram.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Ranking + supersession tuning.
SIMILARITY_WEIGHT = 0.7
STRENGTH_WEIGHT = 0.3
SUPERSEDE_MIN_CONFIDENCE = 0.7
MAX_SUPERSEDED_PER_INSERT = 2
CANDIDATE_K = 6
RECALL_FETCH_BUFFER = 10
DEFAULT_TOKEN_BUDGET = 1500

# answer() never grounds on rejected approaches (and active-only already
# excludes superseded). Allowed grounding types: architecture, convention,
# component, bug_fix, open_thread.
ANSWER_EXCLUDE_TYPES = ["rejected_approach"]


def embedding_text(memory: Memory) -> str:
    """Return the canonical text embedded for `memory`.

    For bug fixes, the symptom/root_cause/fix details are folded in so recall
    can match on them, not just the title/body.
    """
    text = f"{memory.type.value}: {memory.title}\n{memory.body}"
    if memory.type is MemoryType.BUG_FIX and memory.details:
        parts = [memory.details.get(key) for key in ("symptom", "root_cause", "fix")]
        extra = " ".join(p for p in parts if isinstance(p, str) and p)
        if extra:
            text = f"{text}\n{extra}"
    return text


class MemoryEngine:
    """Coordinates extraction, intelligence, the vector store, and metadata."""

    def __init__(
        self,
        client: QwenClient,
        vector_store: ChromaVectorStore,
        metadata_store: SqliteMetadataStore,
        settings: Settings | None,
    ) -> None:
        """Wire the engine to its client and stores and initialise the stores."""
        self._client = client
        self._vectors = vector_store
        self._meta = metadata_store
        self._settings = settings
        self._meta.init()
        self._vectors.init()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MemoryEngine:
        """Build an engine (client + both stores) from `settings`."""
        settings = settings or get_settings()
        client = QwenClient(settings)
        vector_store = ChromaVectorStore(path=settings.chroma_path)
        metadata_store = SqliteMetadataStore(path=settings.sqlite_path)
        return cls(client, vector_store, metadata_store, settings)

    def reset_project(self, project_id: str) -> None:
        """Delete all stored state for `project_id` (metadata, edges, feedback, vectors)."""
        self._meta.reset_project(project_id)
        self._vectors.reset_project(project_id)

    # --- Capture -----------------------------------------------------------
    def remember(
        self,
        content: str,
        project_id: str = "default",
        hint: str | None = None,
        source: str = "manual",
    ) -> list[Memory]:
        """Extract, score, store, then de-conflict memories from `content`.

        All extracted memories are stored first; supersession then runs for each
        with the entire same-batch sibling set excluded, so siblings from one
        remember() call can never supersede each other.
        """
        memories = extract_memories(self._client, content, project_id, source, hint)
        prepared: list[tuple[Memory, list[float]]] = []
        for memory in memories:
            existing = self._meta.find_by_key(project_id, memory.type.value, memory.title)
            if existing is not None:
                memory.id = existing.id

            salience, _rationale = score_memory(self._client, memory)
            memory.salience = salience
            memory.status = "active"

            embedding = self._client.embed([embedding_text(memory)])[0]
            self._meta.upsert_memory(memory)
            self._vectors.add_vector(
                memory.id,
                embedding,
                {"project_id": project_id, "type": memory.type.value},
            )
            prepared.append((memory, embedding))

        sibling_ids = {mem.id for mem, _ in prepared}
        for memory, embedding in prepared:
            self._resolve_supersession(memory, embedding, project_id, exclude_ids=sibling_ids)
            self._meta.upsert_memory(memory)  # persist any status change (e.g. dup-merge)

        return [mem for mem, _ in prepared]

    def _resolve_supersession(
        self,
        memory: Memory,
        embedding: list[float],
        project_id: str,
        exclude_ids: set[str] | None = None,
    ) -> None:
        """Apply guarded, content-driven supersession verdicts for a new memory.

        The new memory may supersede candidates, never the reverse. We exclude
        the memory's own id and all same-batch siblings, require confidence
        >= 0.7, never let a rejected_approach supersede a non-rejected memory,
        protect strictly-higher-salience candidates, treat duplicates as a
        salience-based merge (no edge), and cap supersessions per insert.
        """
        exclude = set(exclude_ids or ())
        exclude.add(memory.id)
        try:
            candidate_hits = self._vectors.query_candidates(embedding, project_id, CANDIDATE_K)
        except Exception:  # noqa: BLE001 - no candidates yet (e.g. empty collection)
            return
        candidate_ids = [cid for cid, _ in candidate_hits if cid not in exclude]
        candidates = self._meta.get_memories(candidate_ids)
        if not candidates:
            return
        by_id = {c.id: c for c in candidates}

        superseded = 0
        for verdict in check_supersession(self._client, memory, candidates):
            if superseded >= MAX_SUPERSEDED_PER_INSERT:
                break
            target = by_id.get(verdict.target_id)
            if target is None or target.id == memory.id:
                continue  # unknown candidate, or self/twin — never supersede
            if verdict.relation == "unrelated":
                continue
            if verdict.confidence < SUPERSEDE_MIN_CONFIDENCE:
                logger.info(
                    "supersession skipped: low confidence %.2f for %s",
                    verdict.confidence, target.id,
                )
                continue

            if verdict.relation in ("supersedes", "contradicts"):
                # Type authority: a rejected_approach can only relate to another
                # rejected_approach — never retire real knowledge.
                if (
                    memory.type is MemoryType.REJECTED_APPROACH
                    and target.type is not MemoryType.REJECTED_APPROACH
                ):
                    logger.info(
                        "supersession blocked: rejected_approach cannot supersede %s",
                        target.id,
                    )
                    continue
                # Directional guard: salience alone protects the candidate.
                if target.salience > memory.salience:
                    logger.info("supersession blocked: protecting higher-salience %s", target.id)
                    continue
                self._meta.set_status(target.id, "superseded")
                self._meta.add_edge(memory.id, target.id, "supersedes")
                superseded += 1
                logger.info(
                    "superseded %s via %s (conf=%.2f)",
                    target.id, verdict.relation, verdict.confidence,
                )
            elif verdict.relation == "duplicate":
                # Merge: keep the higher-salience memory; no edge.
                if target.salience >= memory.salience:
                    memory.status = "superseded"
                    logger.info("duplicate merge: new memory kept as superseded vs %s", target.id)
                    break
                self._meta.set_status(target.id, "superseded")
                superseded += 1
                logger.info("duplicate merge: superseded existing %s in favour of new", target.id)

    # --- Recall ------------------------------------------------------------
    def recall(
        self,
        query: str,
        project_id: str = "default",
        k: int = 5,
        pack: bool = False,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        exclude_types: list[str] | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Rank active memories by 0.7*similarity + 0.3*strength and reinforce them.

        `exclude_types` drops memories of those types. With `pack=True`, also
        returns a budget-bounded packed context.
        """
        excluded = set(exclude_types or [])
        embedding = self._client.embed([query])[0]
        hits = self._vectors.query(
            embedding, k + RECALL_FETCH_BUFFER, where={"project_id": project_id}
        )
        distances = dict(hits)
        records = self._meta.get_memories(list(distances))  # active only

        now = datetime.now(UTC)
        results: list[dict[str, Any]] = []
        for record in records:
            if record.type.value in excluded:
                continue
            similarity = 1.0 - distances.get(record.id, 1.0)
            strength = effective_strength(record, now)
            combined = SIMILARITY_WEIGHT * similarity + STRENGTH_WEIGHT * strength
            results.append(
                {
                    "id": record.id,
                    "type": record.type.value,
                    "title": record.title,
                    "body": record.body,
                    "score": similarity,
                    "strength": strength,
                    "combined": combined,
                }
            )
        results.sort(key=lambda r: r["combined"], reverse=True)
        results = results[:k]

        for result in results:  # reinforce returned memories
            self._meta.update_access(result["id"])

        if pack:
            packed = self._pack(results, token_budget, query)
            return {"results": results, "context": packed.model_dump()}
        return results

    def _pack(
        self, results: list[dict[str, Any]], token_budget: int, query: str
    ) -> PackedContext:
        """Pack `results` into a token-bounded context."""
        return pack(results, token_budget, query, self._client)

    # --- Answer ------------------------------------------------------------
    def answer(
        self,
        question: str,
        project_id: str = "default",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        """Recall, pack, and synthesize a grounded answer to `question`.

        Grounding is built from active memories only, excluding rejected
        approaches (superseded are already excluded). A relevant rejected
        approach, if any, is surfaced as a one-line "Avoid:" note — never as the
        recommended method.
        """
        grounding = self.recall(
            question, project_id=project_id, k=8, exclude_types=ANSWER_EXCLUDE_TYPES
        )
        assert isinstance(grounding, list)  # pack=False path returns a list
        packed = self._pack(grounding, token_budget, question)

        # TODO Phase 3: verify against current code before trusting the fix.
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": build_answer_prompt(question, packed.text)},
        ]
        answer_text = self._client.chat(messages, model=heavy_model_for(self._client))

        rejected = self._relevant_rejected(question, project_id)
        if rejected is not None:
            answer_text = f"{answer_text}\n\nAvoid: {rejected.title} — {rejected.body}"
        return {"answer": answer_text, "used_memory_ids": packed.included_ids}

    def _relevant_rejected(self, question: str, project_id: str) -> Memory | None:
        """Return the most relevant active rejected_approach memory, if any.

        Does not reinforce (it only informs an "Avoid" note).
        """
        embedding = self._client.embed([question])[0]
        hits = self._vectors.query(embedding, CANDIDATE_K, where={"project_id": project_id})
        records = self._meta.get_memories([hit_id for hit_id, _ in hits])
        for record in records:
            if record.type is MemoryType.REJECTED_APPROACH:
                return record
        return None

    # --- Feedback + maintenance -------------------------------------------
    def feedback(self, memory_id: str, helpful: bool) -> Memory | None:
        """Apply a helpful/not-helpful signal to a memory."""
        return apply_feedback(self._meta, memory_id, helpful)

    def maintenance(self, project_id: str = "default") -> dict[str, Any]:
        """Recompute strength and apply decay status transitions (persisted)."""
        now = datetime.now(UTC)
        memories = self._meta.all_memories(project_id)
        transitions = 0
        for memory in memories:
            if memory.status == "superseded":
                continue
            strength = effective_strength(memory, now)
            age_days = (now - memory.created_at).total_seconds() / 86400.0
            new_status = next_status(strength, age_days)
            if new_status != memory.status:
                self._meta.set_status(memory.id, new_status)
                transitions += 1
        return {"project_id": project_id, "scanned": len(memories), "transitions": transitions}

    def stats(self, project_id: str = "default") -> dict[str, Any]:
        """Return memory counts per type for `project_id` (used by `inspect`)."""
        by_type = self._meta.count_by_type(project_id)
        return {
            "project_id": project_id,
            "total": sum(by_type.values()),
            "by_type": by_type,
        }

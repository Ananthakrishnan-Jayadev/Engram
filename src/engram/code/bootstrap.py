"""Bootstrap a project's memory from its code, docs, and git history."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.code.entities import scan_project
from engram.code.git_history import commit_changes, recent_commits

if TYPE_CHECKING:
    from engram.engine import MemoryEngine

# Doc files scanned for architecture/convention memories.
_DOC_FILES = ("README.md", "README.rst", "README.txt", "ARCHITECTURE.md")
_BOOTSTRAP_COMMITS = 20
_MAX_FILES_PER_COMMIT = 10


def _read_docs(project_path: Path) -> str:
    """Concatenate known doc files into one blob (empty if none)."""
    parts: list[str] = []
    for name in _DOC_FILES:
        path = project_path / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            parts.append(f"# {name}\n{text}")
    return "\n\n".join(parts)


def _format_commits(repo: Path, commits: list[dict[str, str]]) -> str:
    """Render commits (subject + changed files + small diff) into one blob."""
    blocks: list[str] = []
    for commit in commits:
        changes = commit_changes(repo, commit["sha"])
        files = ", ".join(changes["files"][:_MAX_FILES_PER_COMMIT])
        blocks.append(
            f"Commit {commit['sha'][:8]} ({commit['date']}): {commit['subject']}\n"
            f"Files: {files}\n{changes['diff']}"
        )
    return "\n\n".join(blocks)


def run_bootstrap(engine: MemoryEngine, project_id: str, project_path: str) -> dict[str, Any]:
    """Build initial memory for a project from entities, docs, and git history.

    Idempotent: entities upsert by key and memories dedup/merge on re-run.
    Returns {entities, memories_by_type, links}.
    """
    root = Path(project_path)

    # 1. Scan + upsert code entities.
    entities = scan_project(root, project_id=project_id)
    for entity in entities:
        engine._meta.upsert_entity(entity)

    # 2. Docs -> architecture/convention memories.
    docs = _read_docs(root)
    if docs.strip():
        engine.remember(
            docs, project_id=project_id, source="bootstrap:docs",
            hint="architecture and conventions",
        )

    # 3. Recent commits -> bug_fix/decision memories.
    commits = recent_commits(root, n=_BOOTSTRAP_COMMITS)
    if commits:
        engine.remember(
            _format_commits(root, commits), project_id=project_id,
            source="bootstrap:git", hint="bug fixes and decisions",
        )

    # 4. Backfill any stashed entity-hint links (remember already links live).
    engine.backfill_entity_links(project_id)

    return {
        "entities": len(entities),
        "memories_by_type": engine._meta.count_by_type(project_id),
        "links": engine._meta.count_links(project_id),
    }

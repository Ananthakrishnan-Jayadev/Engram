"""Code-aware knowledge graph: entity parsing, linking, bootstrap, and sync."""

from engram.code.bootstrap import run_bootstrap
from engram.code.entities import (
    CodeEntity,
    ancestor_keys,
    entity_source,
    match_entities,
    parse_file,
    scan_project,
)
from engram.code.git_history import commit_changes, recent_commits
from engram.code.sync import run_sync

__all__ = [
    "CodeEntity",
    "ancestor_keys",
    "entity_source",
    "match_entities",
    "parse_file",
    "scan_project",
    "commit_changes",
    "recent_commits",
    "run_bootstrap",
    "run_sync",
]

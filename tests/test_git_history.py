"""Tests for git history parsing (subprocess mocked; no real git)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from engram.code.git_history import commit_changes, recent_commits

SEP = "\x1f"


def _completed(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    """Build a fake CompletedProcess-like object."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_recent_commits_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mocked git log is parsed into commit dicts."""
    log = (
        f"sha1{SEP}Alice{SEP}2026-01-01{SEP}fix token bug\n"
        f"sha2{SEP}Bob{SEP}2026-01-02{SEP}add cache layer"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout=log))
    commits = recent_commits("/repo", n=5)
    assert [c["sha"] for c in commits] == ["sha1", "sha2"]
    assert commits[0]["subject"] == "fix token bug"
    assert commits[0]["author"] == "Alice"


def test_non_repo_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero git exit (not a repo) degrades to empty results."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(returncode=128))
    assert recent_commits("/repo") == []
    changes = commit_changes("/repo", "deadbeef")
    assert changes["files"] == []
    assert changes["diff"] == ""


def test_git_missing_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing git binary degrades to empty results."""

    def boom(*args: object, **kwargs: object) -> SimpleNamespace:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert recent_commits("/repo") == []


def test_commit_changes_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """commit_changes splits subject + files and captures the diff."""
    outputs = iter(
        [
            _completed(stdout="fix token bug\n\na.py\nb.py"),
            _completed(stdout="diff --git a/a.py b/a.py\n+guard"),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: next(outputs))
    changes = commit_changes("/repo", "sha1")
    assert changes["subject"] == "fix token bug"
    assert changes["files"] == ["a.py", "b.py"]
    assert "diff --git" in changes["diff"]

"""Read recent git history via the git CLI, degrading gracefully off-repo."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Field/record separators unlikely to appear in commit metadata.
_SEP = "\x1f"
_GIT_TIMEOUT = 15
# Cap a single commit's diff when handing it to the LLM.
DIFF_CHAR_CAP = 1500


def _run_git(repo: str | Path, args: list[str]) -> str | None:
    """Run `git -C <repo> <args>`; return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def recent_commits(repo: str | Path, n: int = 20) -> list[dict[str, str]]:
    """Return up to `n` recent commits as {sha, author, date, subject} dicts.

    Returns [] if `repo` is not a git repository (or git is unavailable).
    """
    fmt = _SEP.join(["%H", "%an", "%ad", "%s"])
    out = _run_git(repo, ["log", f"-n{n}", "--date=short", f"--pretty=format:{fmt}"])
    if not out:
        return []
    commits: list[dict[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split(_SEP)
        if len(parts) < 4:
            continue
        commits.append(
            {"sha": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]}
        )
    return commits


def commit_changes(repo: str | Path, sha: str) -> dict[str, Any]:
    """Return {sha, subject, files, diff} for `sha` (empty fields off-repo)."""
    files_out = _run_git(repo, ["show", "--name-only", "--pretty=format:%s", sha])
    if files_out is None:
        return {"sha": sha, "subject": "", "files": [], "diff": ""}
    lines = files_out.splitlines()
    subject = lines[0] if lines else ""
    files = [line for line in lines[1:] if line.strip()]
    diff_out = _run_git(repo, ["show", "--unified=2", "--pretty=format:", sha]) or ""
    return {"sha": sha, "subject": subject, "files": files, "diff": diff_out[:DIFF_CHAR_CAP]}

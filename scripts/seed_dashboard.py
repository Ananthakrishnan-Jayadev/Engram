"""Seed a persistent "demo" project so the dashboard has real data (live Qwen).

One command fills every dashboard view:
  1. reset the demo project
  2. stage a working copy of the sample project at .demo_project/
  3. bootstrap it (entities + doc memories + links)
  4. remember an original fix, then a superseding refactor (supersedes edges)
  5. edit a function and sync (recheck + flagged memories)

Run after configuring `.env`:  python scripts/seed_dashboard.py
Then:  uvicorn engram.api.app:app --port 8000
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engram.engine import MemoryEngine

PROJECT_ID = "demo"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "eval" / "fixtures" / "sample_project"
WORKDIR = REPO_ROOT / ".demo_project"

README = """\
# Sample Shop

A tiny shop core. `discount.py` exposes `apply_percentage_discount(price, percent)`
which applies a percentage discount to a price. `cart.py` computes checkout totals
on top of it and `inventory.py` tracks stock levels. Convention: prices are floats
rounded to 2 decimal places.
"""

ORIGINAL_FIX = """\
Bug fix in cart.py: checkout() returned totals like 19.0 instead of 19.99 because
the final amount was passed through int(total), truncating the cents. Quick fix:
cast with int(round(total)) so at least whole amounts stay right. Committed as e41f2a.
"""

SUPERSEDING_FIX = """\
Refactor in cart.py: replaced the int(round(total)) hack in checkout() with
round(total, 2), preserving the cents while removing floating-point noise. This
supersedes the earlier int-cast fix for the truncated-cents bug. Committed as 9bc3d77.
"""


def _git(repo: Path, *args: str) -> None:
    """Run a git command in `repo` (UTF-8), printing a warning on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        print(f"  ⚠ git unavailable: {exc}", flush=True)
        return
    if result.returncode != 0:
        print(f"  ⚠ git {' '.join(args)} -> {result.stderr.strip()}", flush=True)


def _init_repo(repo: Path) -> None:
    """Make the staging dir its OWN git repo so bootstrap's history scan sees
    only the sample's commits — never Engram's surrounding repository."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "demo@engram.local")
    _git(repo, "config", "user.name", "Engram Demo")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Initial import of the sample shop modules")


def main() -> None:
    """Build the demo project end to end and print what was created."""
    engine = MemoryEngine.from_settings()
    engine.reset_project(PROJECT_ID)
    print("✅ Engine ready (demo project reset)", flush=True)

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    shutil.copytree(FIXTURE, WORKDIR)
    (WORKDIR / "README.md").write_text(README, encoding="utf-8")
    _init_repo(WORKDIR)
    print(f"✅ Staged sample project (own git repo) at {WORKDIR}", flush=True)

    summary = engine.bootstrap(str(WORKDIR), project_id=PROJECT_ID)
    print(f"✅ Bootstrap: {summary}", flush=True)

    engine.remember(ORIGINAL_FIX, project_id=PROJECT_ID, source="seed")
    engine.remember(SUPERSEDING_FIX, project_id=PROJECT_ID, source="seed")
    print("✅ Remembered original + superseding fixes", flush=True)

    discount = WORKDIR / "discount.py"
    text = discount.read_text(encoding="utf-8")
    discount.write_text(
        text.replace(
            "return price * (percent / 100.0)",
            "return round(price * (1.0 - percent / 100.0), 2)",
        ),
        encoding="utf-8",
    )
    _git(WORKDIR, "add", "-A")
    _git(WORKDIR, "commit", "-q", "-m", "Fix apply_percentage_discount to subtract the discount")
    report = engine.sync_code(str(WORKDIR), project_id=PROJECT_ID)
    print(f"✅ Edited discount.py and synced: {report}", flush=True)

    stats = engine.stats(PROJECT_ID)
    events = engine._meta.list_events(PROJECT_ID, limit=200)
    print(f"\nDemo project ready: {stats['total']} memories {stats['by_type']}", flush=True)
    print(f"Events recorded: {len(events)}", flush=True)
    print("\nNext:  uvicorn engram.api.app:app --port 8000", flush=True)


if __name__ == "__main__":
    main()

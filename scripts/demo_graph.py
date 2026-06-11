"""Code-aware-supersession hero demo (real Qwen + real git).

Walks the full Phase 3 loop on a throwaway copy of the sample project:
  1. reset the demo project for a clean run
  2. copy eval/fixtures/sample_project to a temp dir + git init/commit it
  3. bootstrap -> entities, memories, and memory<->entity links
  4. answer() a question about the code
  5. edit a function + commit, then sync -> show the linked memory rechecked
     and retired/flagged

Run after configuring `.env`:  python scripts/demo_graph.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from engram.engine import MemoryEngine

PROJECT_ID = "demo-graph"
FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "sample_project"

README = """\
# Sample Project

A tiny shop core. `discount.py` exposes `apply_percentage_discount(price, percent)`
which applies a percentage discount to a price. `cart.py` builds on it and
`inventory.py` tracks stock. Convention: prices are floats rounded to 2 dp.
"""


def _hr(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def _git(repo: Path, *args: str) -> None:
    """Run a git command in `repo`, printing a warning on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"  ⚠ git unavailable: {exc}")
        return
    if result.returncode != 0:
        print(f"  ⚠ git {' '.join(args)} -> {result.stderr.strip()}")


def _init_repo(repo: Path) -> None:
    """Initialise a git repo with one commit so history exists."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "demo@engram.local")
    _git(repo, "config", "user.name", "Engram Demo")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Initial import of sample shop modules")


def _print_graph(engine: MemoryEngine) -> None:
    """Print entities, memories, and links for the demo project."""
    entities = engine._meta.list_entities(PROJECT_ID)
    print(f"  entities ({len(entities)}):")
    for entity in entities:
        print(f"    - [{entity.kind}] {entity.entity_key}")

    memories = engine._meta.all_memories(PROJECT_ID)
    print(f"  memories ({len(memories)}):")
    for mem in memories:
        links = engine._meta.entities_for_memory(mem.id)
        link_str = f"  -> {', '.join(links)}" if links else ""
        print(f"    - [{mem.type.value}] {mem.title} ({mem.status}){link_str}")


def main() -> None:
    """Run the code-aware-supersession demo end to end."""
    try:
        engine = MemoryEngine.from_settings()
        engine.reset_project(PROJECT_ID)
        print("✅ Engine constructed (project reset)")
    except Exception as exc:  # noqa: BLE001 - surface setup failures clearly
        print(f"❌ Could not construct engine: {exc}")
        return

    workdir = Path(tempfile.mkdtemp(prefix="engram-demo-"))
    repo = workdir / "sample_project"
    try:
        # 1-2. Copy fixture, add a README, and make it a git repo.
        shutil.copytree(FIXTURE, repo)
        (repo / "README.md").write_text(README, encoding="utf-8")
        _init_repo(repo)
        print(f"✅ Sample project staged + committed at {repo}")

        # 3. Bootstrap.
        _hr("STEP 1 — bootstrap the project")
        summary = engine.bootstrap(str(repo), project_id=PROJECT_ID)
        print(f"  summary: {summary}")
        _print_graph(engine)

        # 4. Answer a question grounded in the code.
        _hr("STEP 2 — answer('how is a percentage discount applied?')")
        out = engine.answer("how is a percentage discount applied?", project_id=PROJECT_ID)
        print(f"  answer:\n    {out['answer']}")
        print(f"  used_memory_ids: {out['used_memory_ids']}")

        # 5. Edit a function, commit, and sync.
        _hr("STEP 3 — edit apply_percentage_discount, commit, then sync")
        discount = repo / "discount.py"
        text = discount.read_text(encoding="utf-8")
        edited = text.replace(
            "return price * (percent / 100.0)",
            "return round(price * (1.0 - percent / 100.0), 2)",
        )
        discount.write_text(edited, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "Fix apply_percentage_discount to subtract the discount")
        print("  edited discount.py and committed the fix")

        report = engine.sync_code(str(repo), project_id=PROJECT_ID)
        print(f"  sync report: {report}")
        _hr("STEP 4 — graph after sync (note superseded/flagged memories)")
        _print_graph(engine)

        print("\n✅ Demo complete.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

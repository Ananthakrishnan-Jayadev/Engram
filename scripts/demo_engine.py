"""End-to-end Phase 2 hero-story demo (uses the live Qwen key).

Walks the full intelligence loop and prints each step clearly:
  1. remember an original bug fix
  2. remember a refactor that supersedes it
  3. show the old memory is now superseded
  4. recall returns only the new fix
  5. answer("how do I handle expired tokens?") synthesizes a flow
  6. feedback(helpful=True) and show the salience/ranking shift

This script is the seed of the demo video — keep the output readable.

Run after configuring `.env`:  python scripts/demo_engine.py
"""

from __future__ import annotations

from engram.engine import MemoryEngine

PROJECT_ID = "demo"

ORIGINAL_FIX = """\
Bug fix in auth.py: /profile returned 500 on expired sessions. get_token()
returns None once a token expires, and we called jwt.decode(token.decode()),
raising AttributeError: 'NoneType' has no attribute 'decode'. Quick fix: wrap
the decode in a try/except AttributeError and return 401 in auth.py.
"""

SUPERSEDING_FIX = """\
Refactor: replaced the try/except hack in auth.py with centralised token
validation. New approach: a require_token() guard validates the token up front
and raises Unauthorized before any decode happens; all routes call it. This
supersedes the earlier try/except AttributeError handling for expired tokens.
"""


def _hr(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main() -> None:
    """Run the hero story end to end."""
    try:
        engine = MemoryEngine.from_settings()
        engine.reset_project(PROJECT_ID)  # clean slate so every run is identical
        print("✅ Engine constructed from settings (project reset)")
    except Exception as exc:  # noqa: BLE001 - surface setup failures clearly
        print(f"❌ Could not construct engine: {exc}")
        return

    # 1. Original fix ---------------------------------------------------
    _hr("STEP 1 — remember the ORIGINAL bug fix")
    original = engine.remember(ORIGINAL_FIX, project_id=PROJECT_ID, source="demo")
    for mem in original:
        print(f"  stored [{mem.type.value}] {mem.title}  (salience={mem.salience:.2f})")
    if not original:
        print("  ❌ nothing extracted; aborting demo")
        return
    original_id = original[0].id

    # 2. Superseding refactor ------------------------------------------
    _hr("STEP 2 — remember the REFACTOR that supersedes it")
    refactor = engine.remember(SUPERSEDING_FIX, project_id=PROJECT_ID, source="demo")
    for mem in refactor:
        print(f"  stored [{mem.type.value}] {mem.title}  (salience={mem.salience:.2f})")

    # 3. Show the old one is superseded --------------------------------
    _hr("STEP 3 — the original memory's status now")
    old = engine._meta.get_memory(original_id)
    if old is not None:
        print(f"  original '{old.title}' -> status = {old.status.upper()}")
        edges = []
        for mem in refactor:
            edges += [(mem.title, dst, kind) for dst, kind in engine._meta.outgoing_edges(mem.id)]
        for src_title, dst, kind in edges:
            print(f"  edge: '{src_title}' --{kind}--> {dst}")

    # 4. Recall returns only the new fix -------------------------------
    _hr("STEP 4 — recall('expired token handling')")
    results = engine.recall("expired token handling", project_id=PROJECT_ID, k=5)
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. ({r['combined']:.3f}) [{r['type']}] {r['title']}")

    # 5. Answer synthesizes a flow -------------------------------------
    _hr("STEP 5 — answer('how do I handle expired tokens?')")
    out = engine.answer("how do I handle expired tokens?", project_id=PROJECT_ID)
    print(f"  answer:\n    {out['answer']}")
    print(f"  used_memory_ids: {out['used_memory_ids']}")

    # 6. Feedback shifts salience --------------------------------------
    _hr("STEP 6 — feedback(helpful=True) on the recalled fix")
    if results:
        target_id = results[0]["id"]
        before = engine._meta.get_memory(target_id)
        engine.feedback(target_id, helpful=True)
        after = engine._meta.get_memory(target_id)
        if before is not None and after is not None:
            print(
                f"  salience: {before.salience:.2f} -> {after.salience:.2f} "
                f"(access_count={after.access_count})"
            )

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()

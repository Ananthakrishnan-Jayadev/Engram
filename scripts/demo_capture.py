"""End-to-end Phase 1 capture demo (uses the live Qwen key).

Remembers a realistic bug-fix session snippet, prints what was extracted, then
recalls it and prints the top results with scores. Each step prints ✅/❌.

Run after configuring `.env`:  python scripts/demo_capture.py
"""

from __future__ import annotations

from engram.engine import MemoryEngine

SESSION_SNIPPET = """\
Debugging session — auth.py

Users hit a 500 when calling /profile with an expired session. Traceback showed
an AttributeError: 'NoneType' object has no attribute 'decode' inside
authenticate(). Root cause: get_token() returns None once a token expires, and
we passed that straight into jwt.decode(token.decode(), ...). The expired-token
branch was never handled.

Fix: added a null-check before decoding —
    token = get_token(request)
    if token is None:
        raise Unauthorized("token expired or missing")
    claims = jwt.decode(token, SECRET, algorithms=["HS256"])

Also noted: we should standardise on raising Unauthorized (not returning None)
across the auth module. Committed as f4a9c21.
"""

PROJECT_ID = "demo"


def main() -> None:
    """Run the remember -> recall capture demo."""
    try:
        engine = MemoryEngine.from_settings()
        print("✅ Engine constructed from settings")
    except Exception as exc:  # noqa: BLE001 - surface setup failures clearly
        print(f"❌ Could not construct engine: {exc}")
        return

    # 1. Remember -------------------------------------------------------
    try:
        stored = engine.remember(SESSION_SNIPPET, project_id=PROJECT_ID, source="demo")
        print(f"✅ Extracted and stored {len(stored)} memories:")
        for mem in stored:
            print(f"   - [{mem.type.value}] {mem.title}")
            if mem.details:
                print(f"       details: {mem.details}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ remember failed: {exc}")
        return

    # 2. Recall ---------------------------------------------------------
    try:
        query = "how did I fix the auth crash"
        results = engine.recall(query, project_id=PROJECT_ID, k=5)
        print(f"\n✅ recall({query!r}) returned {len(results)} results:")
        for rank, result in enumerate(results, start=1):
            print(
                f"   {rank}. ({result['score']:.3f}) "
                f"[{result['type']}] {result['title']}"
            )
            print(f"       {result['body']}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ recall failed: {exc}")
        return

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()

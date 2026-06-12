# Real trace fixture (deferred slot)

This directory is a placeholder for **one hand-curated real development trace** —
an actual sequence of sessions/commits with human-labelled gold answers, stale
memories, and edit-affected memories — to complement the synthetic scenarios.

It is intentionally empty for now (deferred to buffer). When added, a real trace
should mirror the `Scenario` shape produced by `eval/generator.py`:

- ordered events (memories to add; code edits to apply)
- queries with `gold_key` / `stale_keys`
- `supersession_truth` (stale key → superseding key)
- `recheck_truth` (edited entity → affected memory keys)

The synthetic generator is the benchmark backbone; this real trace is the
honesty check that the synthetic labels track reality.

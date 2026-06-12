# Engram Benchmark

The benchmark measures whether Engram's memory actually helps an agent over time
— and, crucially, whether it *forgets correctly*. It is built on **deterministic
synthetic scenarios** as the backbone, with a slot reserved for one hand-curated
real trace ([eval/fixtures/real_trace/](eval/fixtures/real_trace/)).

## Scenarios

[`eval/generator.py`](eval/generator.py) authors a project's evolution
deterministically (`generate_scenario(seed, ...)`, no LLM). Because we *author*
the evolution, every label is exact:

- **events** — an ordered stream of structured memories to add and code edits to
  apply (each edit triggers a sync).
- **queries** — `{text, gold_key, stale_keys}`: the memory that *should* surface
  and the superseded ones that should *not*.
- **supersession_truth** — `stale key → superseding key`.
- **recheck_truth** — `edited entity → memory keys that should be revisited`.
- **checkpoints** — event counts at which the improvement curve is sampled.
- **transcripts** — raw per-memory session text, for end-to-end extraction.

## Strategies (baselines)

All consume the same events/queries ([`eval/baselines.py`](eval/baselines.py)):

- **NoMemory** — returns nothing (floor).
- **NaiveAll** — stores everything, semantic recall only, *never* retires or
  decays (this is the one that keeps surfacing stale answers).
- **Engram** — the full pipeline: salience, decay-aware ranking, content + code
  supersession, packing.

## Metrics

[`eval/metrics.py`](eval/metrics.py):

- **recall_accuracy** — hit@k and MRR of the gold memory.
- **stale_hit_rate** — fraction of queries surfacing a stale memory (lower is
  better; this is where forgetting pays off).
- **forgetting_prf** — precision/recall/F1 of retirements vs `supersession_truth`.
- **recheck_prf** — precision/recall/F1 of flagged/retired vs `recheck_truth`.
- **packing_efficiency** — packed vs naive token cost, plus gold retention.
- **improvement_curve** — recall_accuracy across checkpoints.

## Modes and variance

[`eval/harness.py`](eval/harness.py):

- `run_benchmark(scenario, strategies, mode=...)` — one pass.
  - `mode="direct"` skips extraction by injecting pre-structured memories. It is
    fully deterministic **only with the offline fake client** (used in tests).
    The `scripts/run_benchmark.py --direct` flag uses the **real** engine, so it
    still calls Qwen for salience/supersession — a single fixed-seed run, *not*
    deterministic.
  - `mode="e2e"` drives **real Qwen** extraction / supersession / recheck.
- `run_repeated(scenario, strategy_factory, runs=5)` → `AggregateResults` with
  **mean ± std** per metric. The quality numbers depend on real Qwen judgement,
  so they are reported as mean ± std — that is the honest figure.

Each query also gets a **gold-miss diagnostic** (`gold_miss` in the per-strategy
results): the gold memory's rank, or why it missed — ranked `outside_top_k`, or
`excluded:<status>` (e.g. superseded/forgotten). `--verbose` prints it. Scenario
size is CLI-overridable via `--topics` / `--sessions` (small = quick smoke run).

## "Learned forgetting" — what it actually means

[`eval/tuning.py`](eval/tuning.py) `fit_policy(...)`:

1. Ingest the scenario **once** through the real engine, so real supersession
   decisions are made once and the resulting store is **frozen**.
2. Coordinate-search the **deterministic** policy parameters
   ([`ForgettingPolicy`](src/engram/intelligence/policy.py): per-type half-lives,
   decay thresholds, `w_sim`/`w_str`, packing budget) to maximise
   `objective = recall_accuracy − stale_hit_rate`, **re-ranking the frozen store
   with no further LLM calls**.

So **"learned" = fit to benchmark outcomes**, not a trained model. It is cheap
and fully reproducible (a fixed grid search), instead of re-running Qwen
thousands of times. An *online* feedback-adaptive variant is sketched, off by
default, in [`intelligence/adaptive.py`](src/engram/intelligence/adaptive.py).

Defaults are unchanged: `ForgettingPolicy.default()` is the current hand-set
configuration, so the engine behaves exactly as before unless a fitted policy is
loaded.

## Running

```bash
python scripts/run_benchmark.py --seed 0 --runs 5 --e2e          # quality, mean ± std
python scripts/run_benchmark.py --seed 0 --runs 1 --direct --fit # tune the policy
```

Results are written to `eval/results/latest.json` + `latest.md`; a fitted policy
to `eval/results/policy.json`.

"""Run the Engram benchmark (real Qwen) and optionally fit the forgetting policy.

Usage:
    python scripts/run_benchmark.py --seed 0 --runs 1 --e2e
    python scripts/run_benchmark.py --seed 0 --runs 1 --direct --fit

--e2e drives real extraction/supersession/recheck. --direct skips extraction but
STILL calls Qwen for salience and supersession, so it is a single fixed-seed Qwen
run, not a deterministic one. --fit tunes the deterministic forgetting policy to
the scenario outcomes and writes policy.json.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engram.engine import MemoryEngine  # noqa: E402
from eval.baselines import EngramStrategy, NaiveAll, NoMemory  # noqa: E402
from eval.generator import generate_scenario  # noqa: E402
from eval.harness import AggregateResults, run_benchmark, run_repeated  # noqa: E402
from eval.report import render_console, write_results  # noqa: E402
from eval.tuning import fit_policy  # noqa: E402


def _printer(message: str) -> None:
    """Print a live progress line immediately."""
    print(message, flush=True)


def _print_aggregate(agg: AggregateResults) -> None:
    """Print mean ± std per strategy/metric."""
    print(f"\nAggregate over {agg.runs} run(s), seed={agg.seed}, mode={agg.mode}:", flush=True)
    for name, metrics in agg.per_strategy.items():
        print(f"  {name}", flush=True)
        for path, stats in metrics.items():
            print(f"    {path:<24} {stats['mean']:.3f} ± {stats['std']:.3f}", flush=True)


def main() -> None:
    """Parse args, run the benchmark with live progress, and optionally fit."""
    parser = argparse.ArgumentParser(description="Run the Engram benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--topics", type=int, default=None, help="override scenario topics")
    parser.add_argument("--sessions", type=int, default=None, help="override scenario sessions")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="skip extraction; still a single non-deterministic Qwen run",
    )
    parser.add_argument("--e2e", action="store_true", help="real Qwen extraction (default)")
    parser.add_argument("--fit", action="store_true", help="fit + save the forgetting policy")
    parser.add_argument(
        "--verbose", action="store_true", help="print per-query gold-miss breakdown"
    )
    args = parser.parse_args()

    # Surface the engine's existing INFO logs as live progress, but keep the
    # noisy per-request HTTP client logs out of the way.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    mode = "direct" if args.direct else "e2e"

    scenario_kwargs: dict[str, int] = {}
    if args.topics is not None:
        scenario_kwargs["n_topics"] = args.topics
    if args.sessions is not None:
        scenario_kwargs["sessions"] = args.sessions
    scenario = generate_scenario(args.seed, **scenario_kwargs)
    print(
        f"== Engram benchmark ==  seed={args.seed}  mode={mode}  runs={args.runs}  "
        f"events={len(scenario.events)}  queries={len(scenario.queries)}",
        flush=True,
    )

    workdir = Path(tempfile.mkdtemp(prefix="engram-bench-"))
    proj = workdir / "proj"
    proj.mkdir()
    for path, source in scenario.files.items():
        (proj / path).write_text(source, encoding="utf-8")

    try:
        engine = MemoryEngine.from_settings()

        def factory() -> list:
            return [
                NoMemory(),
                NaiveAll(engine._client.embed),
                EngramStrategy(
                    engine,
                    proj,
                    project_id="bench",
                    mode=mode,
                    transcripts=scenario.transcripts,
                ),
            ]

        _printer(f"Run 1/{args.runs}")
        single = run_benchmark(scenario, factory(), k=5, mode=mode, progress=_printer)

        print("\n" + render_console(single), flush=True)
        json_path, _md_path = write_results(single)
        print(f"\nWrote {json_path}", flush=True)

        if args.verbose:
            breakdown = single.per_strategy.get("engram", {}).get("gold_miss", [])
            print("\nGold-miss breakdown (engram):", flush=True)
            for rec in breakdown:
                rank = "MISSED" if rec["rank"] is None else str(rec["rank"])
                print(
                    f"  rank={rank:<7} {rec['reason']:<22} {rec['gold_key']:<18} | {rec['query']}",
                    flush=True,
                )

        if args.runs > 1:
            print("\nComputing variance across runs...", flush=True)
            agg = run_repeated(scenario, factory, runs=args.runs, k=5, mode=mode, progress=_printer)
            _print_aggregate(agg)

        if args.fit:
            print("\nFitting forgetting policy...", flush=True)
            fitted, report = fit_policy(
                scenario,
                engine,
                proj,
                project_id="fit",
                mode=mode,
                out_path="eval/results/policy.json",
            )
            print(
                f"Fitted policy objective: {report['objective_before']:.3f} -> "
                f"{report['objective_after']:.3f}  (wrote eval/results/policy.json)",
                flush=True,
            )
            print(
                f"  w_sim={fitted.w_sim}  w_str={fitted.w_str}  "
                f"dormant={fitted.dormant_threshold}  budget={fitted.token_budget}",
                flush=True,
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

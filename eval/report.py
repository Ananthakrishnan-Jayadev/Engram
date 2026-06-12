"""Render benchmark Results as a console table, JSON, and a markdown summary."""

from __future__ import annotations

import json
from pathlib import Path

from eval.harness import Results

_RESULTS_DIR = Path("eval/results")

# (column header, dotted path into a strategy's metric dict)
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("hit@k", "recall.hit_at_k"),
    ("mrr", "recall.mrr"),
    ("stale_hit", "stale_hit_rate"),
    ("forget_f1", "forgetting.f1"),
    ("recheck_f1", "recheck.f1"),
)


def _dig(metrics: dict, dotted: str) -> float | None:
    """Walk a dotted path into a nested metric dict."""
    node: object = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, int | float) else None


def _fmt(value: float | None) -> str:
    """Format a metric cell."""
    return "  -  " if value is None else f"{value:.3f}"


def render_table(results: Results) -> str:
    """Return a fixed-width strategies × metrics comparison table."""
    headers = ["strategy", *(h for h, _ in _COLUMNS)]
    name_w = max([len("strategy"), *(len(n) for n in results.per_strategy)])
    widths = [name_w, *(max(len(h), 7) for h, _ in _COLUMNS)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    for name, metrics in results.per_strategy.items():
        cells = [name.ljust(widths[0])]
        for (_, path), w in zip(_COLUMNS, widths[1:], strict=True):
            cells.append(_fmt(_dig(metrics, path)).ljust(w))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def render_console(results: Results) -> str:
    """Return the comparison table plus a headline contrast for stdout."""
    lines = [
        f"Engram benchmark — seed {results.seed} (mode={results.mode}, k={results.k})",
        "",
        render_table(results),
    ]
    engram = results.per_strategy.get("engram")
    naive = results.per_strategy.get("naive_all")
    if engram is not None:
        hit = _fmt(_dig(engram, "recall.hit_at_k"))
        stale = _fmt(_dig(engram, "stale_hit_rate"))
        forget = _fmt(_dig(engram, "forgetting.f1"))
        headline = f"\nHeadline: Engram hit@k={hit}  stale_hit={stale}  forget_f1={forget}"
        if naive is not None:
            headline += f"  (NaiveAll stale_hit={_fmt(_dig(naive, 'stale_hit_rate'))})"
        lines.append(headline)
    return "\n".join(lines)


def write_results(results: Results, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write latest.json and latest.md; return their paths."""
    directory = out_dir or _RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": results.seed,
        "k": results.k,
        "mode": results.mode,
        "strategies": results.per_strategy,
    }
    json_path = directory / "latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = f"# Engram benchmark — seed {results.seed} (mode={results.mode}, k={results.k})\n\n"
    md += "```\n" + render_table(results) + "\n```\n"
    md_path = directory / "latest.md"
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path

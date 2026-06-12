/** Shared color language: one hue per memory type, one per status/event kind. */

export const TYPE_COLORS: Record<string, { dot: string; chip: string; line: string }> = {
  architecture: { dot: "bg-violet-400", chip: "bg-violet-500/15 text-violet-300", line: "#a78bfa" },
  convention: { dot: "bg-sky-400", chip: "bg-sky-500/15 text-sky-300", line: "#38bdf8" },
  component: { dot: "bg-emerald-400", chip: "bg-emerald-500/15 text-emerald-300", line: "#34d399" },
  bug_fix: { dot: "bg-amber-400", chip: "bg-amber-500/15 text-amber-300", line: "#fbbf24" },
  rejected_approach: { dot: "bg-rose-400", chip: "bg-rose-500/15 text-rose-300", line: "#fb7185" },
  open_thread: { dot: "bg-slate-400", chip: "bg-slate-500/15 text-slate-300", line: "#94a3b8" },
};

export const FALLBACK_TYPE = { dot: "bg-neutral-400", chip: "bg-neutral-500/15 text-neutral-300", line: "#a3a3a3" };

export function typeColor(type: string) {
  return TYPE_COLORS[type] ?? FALLBACK_TYPE;
}

export const STATUS_CHIPS: Record<string, string> = {
  active: "bg-emerald-500/15 text-emerald-300",
  superseded: "bg-rose-500/15 text-rose-300 line-through",
  dormant: "bg-amber-500/15 text-amber-300",
  forgotten: "bg-neutral-500/15 text-neutral-400",
};

export const EVENT_STYLE: Record<string, { label: string; chip: string; glyph: string }> = {
  remember: { label: "remember", chip: "bg-emerald-500/15 text-emerald-300", glyph: "+" },
  superseded: { label: "superseded", chip: "bg-rose-500/15 text-rose-300", glyph: "×" },
  supersession_blocked: { label: "blocked", chip: "bg-sky-500/15 text-sky-300", glyph: "⊘" },
  duplicate_merge: { label: "merged", chip: "bg-violet-500/15 text-violet-300", glyph: "≡" },
  flagged: { label: "flagged", chip: "bg-amber-500/15 text-amber-300", glyph: "!" },
  recheck: { label: "recheck", chip: "bg-neutral-500/15 text-neutral-300", glyph: "?" },
};

export const FALLBACK_EVENT = { label: "event", chip: "bg-neutral-500/15 text-neutral-300", glyph: "·" };

export const STRATEGY_COLORS: Record<string, string> = {
  no_memory: "#64748b",
  naive_all: "#fbbf24",
  engram: "#34d399",
};

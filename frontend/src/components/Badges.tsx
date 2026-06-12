import { STATUS_CHIPS, typeColor } from "../theme";

export function TypeChip({ type }: { type: string }) {
  return <span className={`chip ${typeColor(type).chip}`}>{type}</span>;
}

export function StatusChip({ status }: { status: string }) {
  const cls = STATUS_CHIPS[status] ?? "bg-neutral-500/15 text-neutral-300";
  return <span className={`chip ${cls}`}>{status}</span>;
}

export function NeedsUpdateBadge() {
  return (
    <span className="chip bg-amber-500/20 text-amber-300" title="Linked code changed — review">
      needs update
    </span>
  );
}

export function SalienceBar({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="flex items-center gap-2" title={`salience ${value.toFixed(2)}`}>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-ink-700">
        <div className="h-full rounded-full bg-emerald-400/80" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] text-neutral-500">{value.toFixed(2)}</span>
    </div>
  );
}

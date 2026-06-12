import { Handle, Position, type NodeProps } from "reactflow";
import { typeColor } from "../theme";

export interface MemoryNodeData {
  label: string;
  memoryType: string;
  status: string;
  salience: number;
  needsUpdate: boolean;
}

export interface EntityNodeData {
  label: string;
  kind: string;
  path: string;
}

/** A typed-memory node: colored by type, dimmed when superseded, badged when stale. */
export function MemoryNode({ data }: NodeProps<MemoryNodeData>) {
  const superseded = data.status === "superseded";
  const color = typeColor(data.memoryType);
  return (
    <div
      className={`w-64 rounded-lg border bg-ink-900 px-3 py-2 shadow-lg shadow-black/30 transition-opacity ${
        superseded ? "border-ink-700/40 opacity-40" : "border-ink-700"
      }`}
    >
      <Handle type="target" position={Position.Right} className="!bg-neutral-600" />
      <Handle type="source" position={Position.Left} className="!bg-neutral-600" />
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 shrink-0 rounded-full ${color.dot}`} />
        <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
          {data.memoryType}
        </span>
        {data.needsUpdate && (
          <span
            className="ml-auto rounded-full bg-amber-500/20 px-1.5 text-[10px] text-amber-300"
            title="Linked code changed"
          >
            !
          </span>
        )}
      </div>
      <p
        className={`mt-1 text-xs leading-snug text-neutral-200 ${
          superseded ? "line-through decoration-rose-400/60" : ""
        }`}
      >
        {data.label}
      </p>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-ink-700">
        <div
          className="h-full bg-emerald-400/70"
          style={{ width: `${Math.round(data.salience * 100)}%` }}
        />
      </div>
    </div>
  );
}

/** A code-entity node: monospace key, square accent. */
export function EntityNode({ data }: NodeProps<EntityNodeData>) {
  return (
    <div className="rounded-md border border-sky-500/30 bg-ink-850 px-3 py-1.5">
      <Handle type="target" position={Position.Right} className="!bg-sky-600" />
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 shrink-0 rounded-sm bg-sky-400" />
        <span className="font-mono text-[11px] text-sky-200">{data.label}</span>
      </div>
      <p className="mt-0.5 font-mono text-[10px] text-neutral-600">{data.kind}</p>
    </div>
  );
}

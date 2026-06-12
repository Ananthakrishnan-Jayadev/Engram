import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { api, type MemoryDetail, type MemorySummary } from "../api";
import { NeedsUpdateBadge, SalienceBar, StatusChip, TypeChip } from "../components/Badges";
import { Empty, ErrorState, Loading } from "../components/States";
import { useFetch } from "../hooks";
import { useProject } from "../project";

const TYPES = [
  "architecture",
  "convention",
  "component",
  "bug_fix",
  "rejected_approach",
  "open_thread",
];
const STATUSES = ["active", "superseded", "dormant", "forgotten"];

function Select({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1.5 font-mono text-xs text-neutral-300 outline-none focus:border-emerald-500/50"
    >
      <option value="">{placeholder}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function DetailPanel({ pid, memoryId, onClose }: { pid: string; memoryId: string; onClose: () => void }) {
  const [detail, setDetail] = useState<MemoryDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    api
      .memory(pid, memoryId)
      .then(setDetail)
      .catch((e: Error) => setError(e.message));
  }, [pid, memoryId]);

  return (
    <motion.aside
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 24 }}
      transition={{ duration: 0.2 }}
      className="panel sticky top-20 h-fit max-h-[calc(100vh-7rem)] overflow-y-auto p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-sm font-semibold text-neutral-100">{detail?.title ?? "…"}</h2>
        <button
          onClick={onClose}
          className="rounded px-1.5 text-neutral-500 hover:bg-ink-800 hover:text-neutral-300"
        >
          ✕
        </button>
      </div>
      {error && <p className="mt-3 font-mono text-xs text-rose-300">{error}</p>}
      {detail && (
        <div className="mt-3 space-y-4 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <TypeChip type={detail.type} />
            <StatusChip status={detail.status} />
            {detail.needs_update && <NeedsUpdateBadge />}
          </div>
          <p className="leading-relaxed text-neutral-300">{detail.body}</p>
          <SalienceBar value={detail.salience} />

          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wider text-neutral-500">Linked code</p>
            {detail.entities.length ? (
              detail.entities.map((key) => (
                <p key={key} className="font-mono text-xs text-sky-300">
                  {key}
                </p>
              ))
            ) : (
              <p className="text-xs text-neutral-600">none</p>
            )}
          </div>

          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wider text-neutral-500">Supersedes</p>
            {detail.supersedes.length ? (
              detail.supersedes.map((e) => (
                <p key={e.id} className="font-mono text-xs text-rose-300">
                  → {e.id.slice(0, 8)} ({e.kind})
                </p>
              ))
            ) : (
              <p className="text-xs text-neutral-600">nothing</p>
            )}
          </div>

          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wider text-neutral-500">Superseded by</p>
            {detail.superseded_by.length ? (
              detail.superseded_by.map((e) => (
                <p key={e.id} className="font-mono text-xs text-rose-300">
                  ← {e.id.slice(0, 8)} ({e.kind})
                </p>
              ))
            ) : (
              <p className="text-xs text-neutral-600">nothing — current truth</p>
            )}
          </div>

          <p className="font-mono text-[11px] text-neutral-600">
            created {new Date(detail.created_at).toLocaleString()} · accessed{" "}
            {detail.access_count}×
          </p>
        </div>
      )}
    </motion.aside>
  );
}

export default function MemoriesView() {
  const { project } = useProject();
  const [type, setType] = useState("");
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const fetchMemories = useCallback(
    () => api.memories(project, { type: type || undefined, status: status || undefined }),
    [project, type, status],
  );
  const { data, loading, error } = useFetch<MemorySummary[]>(fetchMemories);

  if (loading && !data) return <Loading label="Loading memories…" />;
  if (error) return <ErrorState message={error} />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold text-neutral-100">Memories</h1>
        <span className="font-mono text-[11px] text-neutral-600">{data?.length ?? 0} shown</span>
        <div className="ml-auto flex gap-2">
          <Select value={type} onChange={setType} options={TYPES} placeholder="all types" />
          <Select value={status} onChange={setStatus} options={STATUSES} placeholder="all statuses" />
        </div>
      </div>

      {!data?.length ? (
        <Empty title="No memories match" hint="Clear the filters, or seed the demo project" />
      ) : (
        <div className={`grid gap-4 ${selected ? "lg:grid-cols-[1fr_22rem]" : ""}`}>
          <div className="grid h-fit gap-3 sm:grid-cols-2">
            {data.map((memory) => (
              <motion.button
                key={memory.id}
                layout
                onClick={() => setSelected(memory.id)}
                className={`panel cursor-pointer p-4 text-left transition-colors hover:border-ink-700 ${
                  selected === memory.id ? "border-emerald-500/40" : ""
                } ${memory.status === "superseded" ? "opacity-50" : ""}`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <TypeChip type={memory.type} />
                  <StatusChip status={memory.status} />
                  {memory.needs_update && <NeedsUpdateBadge />}
                </div>
                <p
                  className={`mt-2 text-sm font-medium text-neutral-200 ${
                    memory.status === "superseded" ? "line-through decoration-rose-400/60" : ""
                  }`}
                >
                  {memory.title}
                </p>
                <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-neutral-500">
                  {memory.body}
                </p>
                <div className="mt-3">
                  <SalienceBar value={memory.salience} />
                </div>
              </motion.button>
            ))}
          </div>
          <AnimatePresence>
            {selected && (
              <DetailPanel pid={project} memoryId={selected} onClose={() => setSelected(null)} />
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

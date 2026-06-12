import { AnimatePresence, motion } from "framer-motion";
import { useCallback } from "react";
import { api } from "../api";
import { Empty, ErrorState, Loading } from "../components/States";
import { useFetch } from "../hooks";
import { useProject } from "../project";
import { EVENT_STYLE, FALLBACK_EVENT } from "../theme";

const POLL_MS = 4000;

function timeOf(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
}

export default function EventsView() {
  const { project } = useProject();
  const fetchEvents = useCallback(() => api.events(project, 150), [project]);
  const { data, loading, error } = useFetch(fetchEvents, POLL_MS);

  if (loading && !data) return <Loading label="Loading events…" />;
  if (error && !data) return <ErrorState message={error} />;
  if (!data?.length) {
    return (
      <Empty
        title="No events recorded yet"
        hint="Every remember / supersede / recheck decision lands here as it happens"
      />
    );
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-4 flex items-baseline justify-between">
        <h1 className="text-lg font-semibold text-neutral-100">Decision feed</h1>
        <span className="font-mono text-[11px] text-neutral-600">
          live · polls every {POLL_MS / 1000}s
        </span>
      </div>
      <ol className="space-y-2">
        <AnimatePresence initial={false}>
          {data.map((event) => {
            const style = EVENT_STYLE[event.kind] ?? FALLBACK_EVENT;
            return (
              <motion.li
                key={event.id}
                layout
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="panel flex items-start gap-3 px-4 py-3"
              >
                <span
                  className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-xs ${style.chip}`}
                >
                  {style.glyph}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className={`chip ${style.chip}`}>{style.label}</span>
                    {event.memory_id && (
                      <span className="truncate font-mono text-[11px] text-neutral-500">
                        {event.memory_id.slice(0, 8)}
                      </span>
                    )}
                    <span className="ml-auto shrink-0 font-mono text-[11px] text-neutral-600">
                      {timeOf(event.ts)}
                    </span>
                  </div>
                  <p className="mt-1 break-words text-sm text-neutral-300">{event.detail}</p>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ol>
    </div>
  );
}

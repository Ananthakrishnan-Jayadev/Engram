import { NavLink, Route, Routes } from "react-router-dom";
import { useProject } from "./project";
import DecayView from "./views/DecayView";
import EventsView from "./views/EventsView";
import GraphView from "./views/GraphView";
import MemoriesView from "./views/MemoriesView";
import MetricsView from "./views/MetricsView";

const NAV = [
  { to: "/graph", label: "Graph" },
  { to: "/events", label: "Events" },
  { to: "/memories", label: "Memories" },
  { to: "/metrics", label: "Metrics" },
  { to: "/decay", label: "Decay" },
];

export default function App() {
  const { projects, project, setProject } = useProject();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-ink-700/60 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-semibold tracking-[0.25em] text-emerald-400">
              ENGRAM
            </span>
            <span className="hidden text-[11px] text-neutral-600 sm:block">
              memory that forgets correctly
            </span>
          </div>
          <nav className="flex gap-1">
            {NAV.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm transition-colors ${
                    isActive
                      ? "bg-ink-800 text-neutral-100"
                      : "text-neutral-500 hover:bg-ink-850 hover:text-neutral-300"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[11px] uppercase tracking-wider text-neutral-600">project</span>
            <select
              value={project}
              onChange={(e) => setProject(e.target.value)}
              className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1 font-mono text-xs text-neutral-300 outline-none focus:border-emerald-500/50"
            >
              {(projects.length ? projects : [project]).map((pid) => (
                <option key={pid} value={pid}>
                  {pid}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Routes>
          <Route path="/" element={<GraphView />} />
          <Route path="/graph" element={<GraphView />} />
          <Route path="/events" element={<EventsView />} />
          <Route path="/memories" element={<MemoriesView />} />
          <Route path="/metrics" element={<MetricsView />} />
          <Route path="/decay" element={<DecayView />} />
        </Routes>
      </main>
    </div>
  );
}

/** Typed client for the Engram read API (proxied to :8000 by Vite). */

export interface MemorySummary {
  id: string;
  type: string;
  title: string;
  body: string;
  status: string;
  salience: number;
  created_at: string;
  access_count: number;
  last_accessed: string | null;
  details: Record<string, unknown>;
  needs_update: boolean;
  entities: string[];
}

export interface EdgeRef {
  id: string;
  kind: string;
}

export interface MemoryDetail extends MemorySummary {
  supersedes: EdgeRef[];
  superseded_by: EdgeRef[];
}

export interface GraphNode {
  id: string;
  type: "memory" | "entity";
  position: { x: number; y: number };
  data: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  animated: boolean;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface EventRow {
  id: string;
  ts: string;
  project_id: string;
  kind: string;
  memory_id: string | null;
  detail: string;
}

export interface Stats {
  project_id: string;
  total: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
}

export interface DecayPoint {
  t: string;
  strength: number;
}

export interface DecayCurve {
  id: string;
  title: string;
  type: string;
  status: string;
  points: DecayPoint[];
}

export interface StrategyMetrics {
  recall?: { hit_at_k?: number; mrr?: number };
  stale_hit_rate?: number;
  forgetting?: { precision?: number; recall?: number; f1?: number };
  recheck?: { f1?: number };
  curve?: number[];
  packing?: { token_ratio?: number; gold_retention?: number };
}

export interface MetricsPayload {
  seed?: number;
  k?: number;
  mode?: string;
  strategies?: Record<string, StrategyMetrics>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return (await res.json()) as T;
}

export const api = {
  projects: () => get<string[]>("/api/projects"),
  memories: (pid: string, filters?: { type?: string; status?: string }) => {
    const params = new URLSearchParams();
    if (filters?.type) params.set("type", filters.type);
    if (filters?.status) params.set("status", filters.status);
    const qs = params.toString();
    return get<MemorySummary[]>(`/api/projects/${pid}/memories${qs ? `?${qs}` : ""}`);
  },
  memory: (pid: string, id: string) => get<MemoryDetail>(`/api/projects/${pid}/memories/${id}`),
  graph: (pid: string) => get<GraphPayload>(`/api/projects/${pid}/graph`),
  events: (pid: string, limit = 100) => get<EventRow[]>(`/api/projects/${pid}/events?limit=${limit}`),
  stats: (pid: string) => get<Stats>(`/api/projects/${pid}/stats`),
  metrics: () => get<MetricsPayload>("/api/metrics"),
  decay: (pid: string) => get<DecayCurve[]>(`/api/projects/${pid}/decay`),
};

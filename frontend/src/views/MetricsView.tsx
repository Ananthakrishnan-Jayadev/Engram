import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type MetricsPayload, type StrategyMetrics } from "../api";
import { Empty, ErrorState, Loading } from "../components/States";
import { useFetch } from "../hooks";
import { STRATEGY_COLORS } from "../theme";

const TOOLTIP_STYLE = {
  backgroundColor: "#15151f",
  border: "1px solid #262635",
  borderRadius: 8,
  fontSize: 12,
};

function barData(strategies: Record<string, StrategyMetrics>) {
  const value = (s: StrategyMetrics, metric: string): number => {
    if (metric === "hit@k") return s.recall?.hit_at_k ?? 0;
    if (metric === "stale_hit") return s.stale_hit_rate ?? 0;
    return s.forgetting?.f1 ?? 0;
  };
  return ["hit@k", "stale_hit", "forget_f1"].map((metric) => ({
    metric,
    ...Object.fromEntries(
      Object.entries(strategies).map(([name, s]) => [name, value(s, metric)]),
    ),
  }));
}

function curveData(strategies: Record<string, StrategyMetrics>) {
  const longest = Math.max(0, ...Object.values(strategies).map((s) => s.curve?.length ?? 0));
  return Array.from({ length: longest }, (_, i) => ({
    checkpoint: `cp${i + 1}`,
    ...Object.fromEntries(
      Object.entries(strategies).map(([name, s]) => [name, s.curve?.[i] ?? null]),
    ),
  }));
}

function Card({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="panel px-4 py-3">
      <p className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</p>
      <p className="mt-1 font-mono text-2xl text-neutral-100">{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-neutral-600">{hint}</p>}
    </div>
  );
}

export default function MetricsView() {
  const { data, loading, error } = useFetch<MetricsPayload>(api.metrics);

  if (loading) return <Loading label="Loading metrics…" />;
  if (error) return <ErrorState message={error} />;
  const strategies = data?.strategies;
  if (!strategies || !Object.keys(strategies).length) {
    return (
      <Empty
        title="No benchmark results yet"
        hint="python scripts/run_benchmark.py --seed 0 --runs 1 writes eval/results/latest.json"
      />
    );
  }

  const names = Object.keys(strategies);
  const packing = strategies.engram?.packing;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <div className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold text-neutral-100">Benchmark</h1>
        <span className="font-mono text-[11px] text-neutral-600">
          seed {data?.seed ?? "?"} · mode {data?.mode ?? "?"} · k={data?.k ?? "?"}
        </span>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel p-4">
          <h2 className="mb-3 text-sm font-medium text-neutral-300">
            Strategy comparison <span className="text-neutral-600">(stale_hit: lower is better)</span>
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData(strategies)} barGap={3}>
              <CartesianGrid stroke="#262635" vertical={false} />
              <XAxis dataKey="metric" stroke="#71717a" fontSize={12} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 1]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#1b1b2755" }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {names.map((name) => (
                <Bar
                  key={name}
                  dataKey={name}
                  fill={STRATEGY_COLORS[name] ?? "#a3a3a3"}
                  radius={[3, 3, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel p-4">
          <h2 className="mb-3 text-sm font-medium text-neutral-300">
            Improvement curve <span className="text-neutral-600">(hit@k per checkpoint)</span>
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={curveData(strategies)}>
              <CartesianGrid stroke="#262635" vertical={false} />
              <XAxis dataKey="checkpoint" stroke="#71717a" fontSize={12} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 1]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {names.map((name) => (
                <Line
                  key={name}
                  dataKey={name}
                  stroke={STRATEGY_COLORS[name] ?? "#a3a3a3"}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card
          label="Engram stale hits"
          value={(strategies.engram?.stale_hit_rate ?? 0).toFixed(3)}
          hint="fraction of queries surfacing a superseded memory"
        />
        <Card
          label="Packing token ratio"
          value={packing?.token_ratio != null ? packing.token_ratio.toFixed(2) : "—"}
          hint="packed context vs naive top-k dump"
        />
        <Card
          label="Gold retention in pack"
          value={packing?.gold_retention != null ? packing.gold_retention.toFixed(2) : "—"}
          hint="gold memories that survive packing"
        />
      </div>
    </motion.div>
  );
}

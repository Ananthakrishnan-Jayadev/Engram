import { motion } from "framer-motion";
import { useCallback, useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type DecayCurve } from "../api";
import { Empty, ErrorState, Loading } from "../components/States";
import { useFetch } from "../hooks";
import { useProject } from "../project";
import { typeColor } from "../theme";

// STRETCH VIEW: strength-over-time curves from /decay.
const MAX_CURVES = 8;

interface Row {
  t: number;
  [memoryId: string]: number;
}

/** Merge per-memory point series into one row-per-timestamp table. */
function toRows(curves: DecayCurve[]): Row[] {
  const byTime = new Map<number, Row>();
  for (const curve of curves) {
    for (const point of curve.points) {
      const t = new Date(point.t).getTime();
      const row = byTime.get(t) ?? { t };
      row[curve.id] = point.strength;
      byTime.set(t, row);
    }
  }
  return [...byTime.values()].sort((a, b) => a.t - b.t);
}

export default function DecayView() {
  const { project } = useProject();
  const fetchDecay = useCallback(() => api.decay(project), [project]);
  const { data, loading, error } = useFetch(fetchDecay);

  const curves = useMemo(() => (data ?? []).slice(0, MAX_CURVES), [data]);
  const rows = useMemo(() => toRows(curves), [curves]);

  if (loading) return <Loading label="Loading decay curves…" />;
  if (error) return <ErrorState message={error} />;
  if (!curves.length) return <Empty title="No active memories to decay" />;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <div className="flex items-baseline gap-3">
        <h1 className="text-lg font-semibold text-neutral-100">Decay</h1>
        <span className="font-mono text-[11px] text-neutral-600">
          effective strength, creation → +30 days
        </span>
      </div>

      <div className="panel p-4">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={rows}>
            <CartesianGrid stroke="#262635" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t: number) => new Date(t).toLocaleDateString()}
              stroke="#71717a"
              fontSize={11}
              tickLine={false}
            />
            <YAxis domain={[0, 1]} stroke="#71717a" fontSize={11} tickLine={false} />
            <Tooltip
              labelFormatter={(t) => new Date(Number(t)).toLocaleString()}
              contentStyle={{
                backgroundColor: "#15151f",
                border: "1px solid #262635",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            {curves.map((curve) => (
              <Line
                key={curve.id}
                dataKey={curve.id}
                name={curve.title}
                stroke={typeColor(curve.type).line}
                strokeWidth={1.8}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex flex-wrap gap-x-5 gap-y-1">
        {curves.map((curve) => (
          <span key={curve.id} className="flex items-center gap-1.5 text-xs text-neutral-400">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: typeColor(curve.type).line }}
            />
            {curve.title}
          </span>
        ))}
      </div>
    </motion.div>
  );
}

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ScalarPoint } from "../../api/types";
import { formatNumber } from "../../lib/format";

const SERIES_COLORS = ["#5eb0ff", "#f87171", "#34d399", "#fbbf24", "#a78bfa", "#22d3ee"];

export interface MetricSeries {
  tag: string;
  label: string;
  points: ScalarPoint[];
}

export default function MetricChart({
  title,
  series,
  onRemove,
}: {
  title: string;
  series: MetricSeries[];
  onRemove?: () => void;
}) {
  const stepSet = new Set<number>();
  for (const s of series) for (const [step] of s.points) stepSet.add(step);
  const steps = [...stepSet].sort((a, b) => a - b);
  const byTag = series.map((s) => new Map(s.points.map((p) => [p[0], p[2]])));
  const data = steps.map((step) => {
    const row: Record<string, number | null> = { step };
    series.forEach((s, index) => {
      row[s.tag] = byTag[index].get(step) ?? null;
    });
    return row;
  });
  const empty = data.length === 0;

  return (
    <div className="panel">
      <div className="flex items-center justify-between border-b border-ink-700 px-4 py-2">
        <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
        </div>
        <div className="flex items-center gap-3">
          {series.length > 1 &&
            series.map((s, index) => (
              <span key={s.tag} className="flex items-center gap-1 text-[11px] text-slate-500">
                <span
                  className="h-1.5 w-3 rounded-sm"
                  style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
                />
                {s.label}
              </span>
            ))}
          {onRemove && (
            <button
              onClick={onRemove}
              className="text-slate-600 transition-colors hover:text-slate-300"
              title="Remove chart"
            >
              ✕
            </button>
          )}
        </div>
      </div>
      <div className="h-48 px-2 py-2">
        {empty ? (
          <div className="flex h-full items-center justify-center text-xs text-slate-600">
            no data yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1c2333" strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="step"
                stroke="#39445c"
                tick={{ fill: "#64748b", fontSize: 10 }}
                tickFormatter={(value) => formatNumber(value, 0)}
                tickLine={false}
              />
              <YAxis
                stroke="#39445c"
                tick={{ fill: "#64748b", fontSize: 10 }}
                tickFormatter={(value) => formatNumber(value, 1)}
                tickLine={false}
                width={48}
              />
              <Tooltip
                contentStyle={{
                  background: "#0f131d",
                  border: "1px solid #273044",
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#94a3b8" }}
                labelFormatter={(value) => `step ${formatNumber(Number(value), 0)}`}
                formatter={(value: number, name: string) => [
                  formatNumber(value, 2),
                  series.find((s) => s.tag === name)?.label ?? name,
                ]}
              />
              {series.map((s, index) => (
                <Line
                  key={s.tag}
                  type="monotone"
                  dataKey={s.tag}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

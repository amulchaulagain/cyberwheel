import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  useEvalSummaries,
  useMetricsSummaries,
  useRunRecords,
  useScalarsFor,
} from "../api/hooks";
import type { EvalSummary, RunRecord, StatBlock } from "../api/types";
import PageHeader from "../components/PageHeader";
import MetricChart from "../components/charts/MetricChart";
import StatusBadge from "../components/runs/StatusBadge";
import { formatNumber } from "../lib/format";

function shortTag(tag: string): string {
  return tag.split("/").slice(1).join("/") || tag;
}

function formatValue(value: unknown): string {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function StatCell({ stat }: { stat: StatBlock | undefined }) {
  if (!stat || stat.mean === null)
    return <span className="text-slate-600">—</span>;
  const half =
    stat.ci95_hi !== null && stat.ci95_lo !== null
      ? (stat.ci95_hi - stat.ci95_lo) / 2
      : null;
  return (
    <span>
      {formatNumber(stat.mean, 1)}
      {half !== null && half > 0 && (
        <span className="text-slate-500"> ± {formatNumber(half, 1)}</span>
      )}
      <span className="ml-1 text-[10px] text-slate-600">n={stat.n}</span>
    </span>
  );
}

export default function ComparePage() {
  const [searchParams] = useSearchParams();
  const ids = useMemo(
    () => (searchParams.get("runs") ?? "").split(",").filter(Boolean),
    [searchParams],
  );
  const [showAllParams, setShowAllParams] = useState(false);

  const recordQueries = useRunRecords(ids);
  const runs = recordQueries
    .map((query) => query.data)
    .filter((run): run is RunRecord => Boolean(run));
  const failed = ids.filter(
    (_, index) => recordQueries[index]?.isError,
  );

  const trainRuns = runs.filter((run) => run.kind === "train");
  const evalRuns = runs.filter((run) => run.kind === "evaluate");

  // Training curves: union of episodic-return tags (+ SPS) across the train
  // runs; each run fetches only the tags it actually has.
  const trainIds = trainRuns.map((run) => run.id);
  const metricsQueries = useMetricsSummaries(trainIds);
  const tagsByRun = useMemo(
    () =>
      trainIds.map((_, index) =>
        Object.values(metricsQueries[index]?.data?.tags ?? {}).flat(),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [metricsQueries.map((query) => query.dataUpdatedAt).join(",")],
  );
  const chartTags = useMemo(() => {
    const union = new Set<string>();
    for (const tags of tagsByRun)
      for (const tag of tags)
        if (
          (tag.startsWith("charts/") && tag.endsWith("_episodic_return")) ||
          tag === "charts/SPS"
        )
          union.add(tag);
    return [...union].sort();
  }, [tagsByRun]);
  const scalarQueries = useScalarsFor(
    trainIds.map((runId, index) => ({
      runId,
      tags: chartTags.filter((tag) => tagsByRun[index]?.includes(tag)),
    })),
  );

  const evalIds = evalRuns.map((run) => run.id);
  const summaryQueries = useEvalSummaries(evalIds);
  const summaries: (EvalSummary | undefined)[] = summaryQueries.map(
    (query) => query.data,
  );
  const evalMetricNames = useMemo(() => {
    const union = new Set<string>();
    for (const summary of summaries)
      for (const metric of summary?.metrics ?? []) union.add(metric);
    return [...union].sort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summaryQueries.map((query) => query.dataUpdatedAt).join(",")]);

  // Config diff across all selected runs.
  const paramRows = useMemo(() => {
    const keys = new Set<string>();
    for (const run of runs) for (const key of Object.keys(run.params)) keys.add(key);
    const rows = [...keys].sort().map((key) => {
      const values = runs.map((run) => formatValue(run.params[key]));
      return { key, values, differs: new Set(values).size > 1 };
    });
    const pseudo = (["base_config", "experiment_name", "checkpoint"] as const).map(
      (key) => {
        const values = runs.map((run) => formatValue(run[key]));
        return { key, values, differs: new Set(values).size > 1 };
      },
    );
    return [...pseudo, ...rows];
  }, [runs]);
  const visibleParamRows = showAllParams
    ? paramRows
    : paramRows.filter((row) => row.differs);

  if (ids.length < 2) {
    return (
      <div>
        <PageHeader title="Compare runs" subtitle="Select at least two runs" />
        <div className="p-8 text-sm text-slate-400">
          Pick two or more runs on the <Link className="text-sky-400" to="/">dashboard</Link>{" "}
          and hit “Compare”.
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Compare runs"
        subtitle={`${runs.length} run${runs.length === 1 ? "" : "s"} side by side`}
      />
      <div className="space-y-6 p-8">
        {/* Run chips */}
        <div className="flex flex-wrap gap-2">
          {runs.map((run) => (
            <Link
              key={run.id}
              to={`/runs/${run.kind}/${run.id}`}
              className="panel flex items-center gap-2 px-3 py-1.5 text-sm text-slate-200 transition-colors hover:bg-ink-800/70"
            >
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
                  run.kind === "train"
                    ? "bg-indigo-950/70 text-indigo-300"
                    : "bg-cyan-950/70 text-cyan-300"
                }`}
              >
                {run.kind}
              </span>
              {run.display_name}
              <StatusBadge status={run.status} />
            </Link>
          ))}
          {failed.map((id) => (
            <div
              key={id}
              className="panel px-3 py-1.5 font-mono text-sm text-rose-400"
              title="run not found"
            >
              {id} — not found
            </div>
          ))}
        </div>

        {/* Training metric overlays */}
        {trainRuns.length > 0 && (
          <div className="space-y-3">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Training metrics
            </div>
            {chartTags.length === 0 ? (
              <div className="panel px-4 py-6 text-center text-xs text-slate-600">
                no metrics recorded yet
              </div>
            ) : (
              <div className="grid gap-4 xl:grid-cols-2">
                {chartTags.map((tag) => (
                  <MetricChart
                    key={tag}
                    title={shortTag(tag)}
                    series={trainRuns
                      .map((run, index) => ({
                        tag: `${run.id}:${tag}`,
                        label: run.display_name,
                        points:
                          scalarQueries[index]?.data?.series?.[tag] ?? [],
                      }))
                      .filter((series) => series.points.length > 0)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Evaluation summary comparison */}
        {evalRuns.length > 0 && (
          <div className="space-y-3">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Evaluation summaries · mean ± 95% CI (Student-t)
            </div>
            <div className="panel overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2 font-medium">Metric</th>
                    {evalRuns.map((run) => (
                      <th key={run.id} className="px-3 py-2 font-medium">
                        {run.display_name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-ink-800 text-slate-400">
                    <td className="px-3 py-1.5">seeds × episodes</td>
                    {evalRuns.map((run, index) => {
                      const summary = summaries[index];
                      return (
                        <td key={run.id} className="px-3 py-1.5 font-mono">
                          {summary
                            ? `${summary.seeds.length} × ${summary.num_episodes}`
                            : "—"}
                        </td>
                      );
                    })}
                  </tr>
                  {evalMetricNames.map((metric) => (
                    <tr
                      key={metric}
                      className="border-b border-ink-800 font-mono text-slate-300 last:border-0"
                    >
                      <td className="px-3 py-1.5">{metric}</td>
                      {evalRuns.map((run, index) => (
                        <td key={run.id} className="px-3 py-1.5">
                          <StatCell stat={summaries[index]?.overall?.[metric]} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Config diff */}
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Configuration
            </div>
            <button
              className="btn-ghost !px-2 !py-0.5 !text-[11px] text-slate-500"
              onClick={() => setShowAllParams((previous) => !previous)}
            >
              {showAllParams ? "differences only" : "show all"}
            </button>
          </div>
          <div className="panel overflow-x-auto">
            {visibleParamRows.length === 0 ? (
              <div className="px-4 py-6 text-center text-xs text-slate-600">
                no configuration differences
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2 font-medium">Parameter</th>
                    {runs.map((run) => (
                      <th key={run.id} className="px-3 py-2 font-medium">
                        {run.display_name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visibleParamRows.map((row) => (
                    <tr
                      key={row.key}
                      className={`border-b border-ink-800 last:border-0 ${
                        row.differs ? "bg-amber-950/10" : ""
                      }`}
                    >
                      <td className="px-3 py-1.5 font-mono text-slate-400">
                        {row.key}
                      </td>
                      {row.values.map((value, index) => (
                        <td
                          key={index}
                          className={`max-w-64 truncate px-3 py-1.5 font-mono ${
                            row.differs ? "text-amber-200" : "text-slate-300"
                          }`}
                          title={value}
                        >
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

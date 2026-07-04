import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import {
  useCheckpoints,
  useLogs,
  useMetricsSummary,
  useRun,
  useScalars,
  useStopRun,
} from "../api/hooks";
import PageHeader from "../components/PageHeader";
import MetricChart from "../components/charts/MetricChart";
import LogTail from "../components/logs/LogTail";
import ProgressBar from "../components/runs/ProgressBar";
import StatusBadge from "../components/runs/StatusBadge";
import { estimateEta, formatDuration, formatNumber } from "../lib/format";

function shortTag(tag: string): string {
  return tag.split("/").slice(1).join("/") || tag;
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-lg text-slate-100">{children}</div>
    </div>
  );
}

export default function TrainingRunPage() {
  const { runId } = useParams();
  const { data: run, error } = useRun(runId);
  const active = run?.status === "running" || run?.status === "queued";
  const stop = useStopRun();

  const { data: summary } = useMetricsSummary(runId, Boolean(active));
  const { data: checkpoints } = useCheckpoints(run?.status !== "queued" ? runId : undefined);
  const { data: logs } = useLogs(runId, Boolean(active));

  const allTags = useMemo(
    () => Object.values(summary?.tags ?? {}).flat(),
    [summary],
  );
  const returnTags = allTags.filter((tag) => tag.endsWith("_episodic_return") && tag.startsWith("charts/"));
  const [extraTags, setExtraTags] = useState<string[]>([]);
  const pinnedTags = [...returnTags, "charts/SPS"];
  const chartTags = [...new Set([...pinnedTags, ...extraTags])].filter((tag) =>
    allTags.includes(tag),
  );
  const { data: scalars } = useScalars(runId, chartTags, Boolean(active));
  const series = scalars?.series ?? {};

  if (error) {
    return (
      <div className="p-8 text-sm text-red-400">{(error as Error).message}</div>
    );
  }
  if (!run) return <div className="p-8 text-sm text-slate-500">Loading…</div>;

  const totalTimesteps = Number(run.params.total_timesteps ?? 0);
  const step = summary?.last_step ?? run.last_global_step ?? 0;
  const progress =
    run.status === "succeeded" ? 1 : totalTimesteps ? step / totalTimesteps : null;
  const availableExtra = allTags.filter((tag) => !chartTags.includes(tag));

  return (
    <div>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            {run.display_name} <StatusBadge status={run.status} />
          </span>
        }
        subtitle={
          <span className="font-mono">
            {run.id} · {run.base_config}
          </span>
        }
        actions={
          <>
            {checkpoints?.checkpoints.length ? (
              <Link to={`/evaluate/new?source=${run.id}`} className="btn-ghost">
                Evaluate
              </Link>
            ) : null}
            {active && (
              <button
                className="btn-danger"
                disabled={stop.isPending}
                onClick={() => stop.mutate(run.id)}
              >
                Stop run
              </button>
            )}
          </>
        }
      />

      <div className="space-y-5 p-8">
        <div className="panel flex flex-wrap items-center gap-x-10 gap-y-4 px-6 py-4">
          <div className="min-w-56 flex-1">
            <div className="mb-1.5 flex justify-between text-[11px] text-slate-500">
              <span className="font-medium uppercase tracking-wider">Progress</span>
              <span className="font-mono">
                {formatNumber(step, 0)} / {formatNumber(totalTimesteps, 0)} steps
              </span>
            </div>
            <ProgressBar value={progress} />
          </div>
          <Stat label="SPS">{formatNumber(run.last_sps ?? null, 0)}</Stat>
          <Stat label="Elapsed">{formatDuration(run.started_at, run.ended_at)}</Stat>
          {active && <Stat label="ETA">{estimateEta(progress, run.started_at)}</Stat>}
          <Stat label="Networks">
            {Array.isArray(run.params.network_config)
              ? (run.params.network_config as string[]).length
              : 1}
          </Stat>
          <Stat label="Envs">{String(run.params.num_envs ?? "—")}</Stat>
          <Stat label="Seed">{String(run.params.seed ?? "—")}</Stat>
        </div>

        {run.status === "failed" && (
          <div className="rounded-md border border-red-900/60 bg-red-950/40 px-4 py-2.5 text-sm text-red-300">
            Run failed with exit code {run.exit_code}. Check the logs below.
          </div>
        )}

        <div>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300">Metrics</h2>
            {availableExtra.length > 0 && (
              <select
                className="field-input !w-56 !py-1 text-xs"
                value=""
                onChange={(event) => {
                  if (event.target.value)
                    setExtraTags((previous) => [...previous, event.target.value]);
                }}
              >
                <option value="">+ add chart…</option>
                {availableExtra.map((tag) => (
                  <option key={tag} value={tag}>
                    {tag}
                  </option>
                ))}
              </select>
            )}
          </div>
          {chartTags.length === 0 ? (
            <div className="panel px-4 py-10 text-center text-sm text-slate-500">
              Waiting for the first metrics…
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              {returnTags.length > 0 && (
                <MetricChart
                  title="Episodic return"
                  series={returnTags.map((tag) => ({
                    tag,
                    label: shortTag(tag).replace("_episodic_return", ""),
                    points: series[tag] ?? [],
                  }))}
                />
              )}
              {allTags.includes("charts/SPS") && (
                <MetricChart
                  title="Throughput (steps/s)"
                  series={[{ tag: "charts/SPS", label: "SPS", points: series["charts/SPS"] ?? [] }]}
                />
              )}
              {extraTags.map((tag) => (
                <MetricChart
                  key={tag}
                  title={tag}
                  series={[{ tag, label: shortTag(tag), points: series[tag] ?? [] }]}
                  onRemove={() =>
                    setExtraTags((previous) => previous.filter((t) => t !== tag))
                  }
                />
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <div className="panel">
            <div className="border-b border-ink-700 px-4 py-2.5 text-xs font-medium uppercase tracking-wider text-slate-500">
              Checkpoints
            </div>
            {checkpoints?.checkpoints.length ? (
              <div className="divide-y divide-ink-700/70">
                {checkpoints.checkpoints.map((tag) => (
                  <div key={String(tag)} className="flex items-center justify-between px-4 py-2">
                    <span className="font-mono text-sm text-slate-300">
                      {tag === "agent"
                        ? "latest (agent.pt)"
                        : `step ${formatNumber(Number(tag), 0)}`}
                    </span>
                    <Link
                      to={`/evaluate/new?source=${run.id}&checkpoint=${tag}`}
                      className="btn-ghost !px-2 !py-1 !text-xs"
                    >
                      Evaluate
                    </Link>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-xs text-slate-600">
                No checkpoints saved yet
              </div>
            )}
          </div>

          <div className="panel">
            <div className="border-b border-ink-700 px-4 py-2.5 text-xs font-medium uppercase tracking-wider text-slate-500">
              Configuration snapshot
            </div>
            <pre className="max-h-72 overflow-auto px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400">
              {JSON.stringify(run.params, null, 2)}
            </pre>
          </div>
        </div>

        <div className="panel">
          <div className="border-b border-ink-700 px-4 py-2.5 text-xs font-medium uppercase tracking-wider text-slate-500">
            Log output
          </div>
          <LogTail content={logs?.content ?? ""} />
        </div>
      </div>
    </div>
  );
}
